#!/usr/bin/env python3

import argparse
import functools
import io
import json
import numpy
import sklearn
import sklearn.metrics
import sys


class Char(str):
	@functools.wraps(str.__new__)
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if len(new) != 1:
			raise ValueError("Char must be a string of length 1, got %d"
				% len(new))
		return new


class Fraction(float):
	@functools.wraps(float.__new__)
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if (new < 0) or (new > 1.0):
			raise ValueError("Fraction must be in [0, 1], got %f" % new)
		return new


class PositiveInt(int):
	@functools.wraps(int.__new__)
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if new <= 0:
			raise ValueError("PositiveInt cannot be %d" % new)
		return new


def get_fp(fp_or_fn, *ka, factory=open, **kw):
	"""
	wrapper to open()-like file handle factory;
	tends to be type-safe when passed with an already-opened file handle
	"""
	if isinstance(fp_or_fn, io.IOBase):
		return fp_or_fn
	elif isinstance(fp_or_fn, str):
		return factory(fp_or_fn, *ka, **kw)
	else:
		raise TypeError("the first argument must be str or io.IOBase, not '%s'"
			% type(fp_or_fn).__name__)
	return


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type=str, nargs="?", default="-",
		help="input simulation outputs acquired from kdiv_sampling.py, in "
			" .json format (default: read from stdin)")
	ap.add_argument("-o", "--output", type=str, metavar="tsv",
		default="-",
		help="output kernel divergence table in tsv format (default: stdout)")
	ap.add_argument("-d", "--delimiter", type=Char, metavar="table",
		default=Char("\t"),
		help="deliminter used in the output table (default: <tab>)")
	# parameters
	ag = ap.add_mutually_exclusive_group(required=True)
	ag.add_argument("-p", "--eigval-preserve-frac", type=Fraction,
		metavar="float",
		help="preserved fraction of eigenvalues for noise filtering "
			"(exclusive with --eigval-preserve-num)")
	ag.add_argument("-n", "--eigval-preserve-num", type=PositiveInt,
		metavar="int",
		help="number of preserved largest eigenvalues for noise filtering "
			"(exclusive with --eigval-preserve-frac)")
	ap.add_argument("--compare-to-last", action="store_true",
		help="compare each step to the last (largest) simulated sampling "
			"depth rather than between steps; this calculates the kernel "
			"divergence between the subset to the dataset (default: off)")
	# parse and refine args
	args = ap.parse_args()
	if args.input == "-":
		args.input = sys.stdin
	if args.output == "-":
		args.output = sys.stdout
	return args


def load_sampling_results(file):
	with get_fp(file, "r") as fp:
		smpl_res = json.load(fp)
	# make eigvals lists into numpy arrays for performance
	return smpl_res


def inplace_filter_eigvals_by_frac(eigvals, frac):
	# the argmax() founds the first element index which the cumu-sum reached
	# eigv_presv threshold, then we set all elements after this to zero
	cutoff_index = (numpy.cumsum(eigvals) >= frac).argmax()
	eigvals[cutoff_index + 1:] = 0.0
	return eigvals


def inplace_filter_eigvals_by_num(eigvals, num):
	eigvals[num:] = 0.0
	return eigvals


def normalize_eigevals(eigvals_list, eigv_pfrac=None, eigv_pnum=None)\
		-> numpy.ndarray:
	eigvals = numpy.asarray(eigvals_list)
	eigvals /= eigvals.sum()  # normalize to sum = 1.0
	if eigv_pfrac is not None:
		return inplace_filter_eigvals_by_frac(eigvals, eigv_pfrac)
	elif eigv_pnum is not None:
		return inplace_filter_eigvals_by_num(eigvals, eigv_pnum)
	else:
		raise ValueError("either eigv_pfrac and eigv_pnum must be provided")


def calc_kernel_divergence(norm_eigvals_1, norm_eigvals_2):
	# ensure lenth vec2 >= vec1
	if len(norm_eigvals_1) > len(norm_eigvals_2):
		return calc_kernel_divergence(norm_eigvals_2, norm_eigvals_1)
	# ensure all values are non-negative
	# this is theoretically true as rbf kernel is semi-positive definite
	# may have negative values close to 0 as a results of imperfect precision
	norm_eigvals_diff = norm_eigvals_2.copy()
	norm_eigvals_diff[:len(norm_eigvals_1)] -= norm_eigvals_1
	# return the infinity norm
	return numpy.abs(norm_eigvals_diff).max()


def calc_simulation_kernel_divergence(simulation_res, *,
		eigv_pfrac=None, eigv_pnum=None, compare_to_last=False) -> list:
	norm_eigvals = [normalize_eigevals(i, eigv_pfrac, eigv_pnum)
		for i in simulation_res]
	kdiv_list = list()
	if compare_to_last:
		for eigv in norm_eigvals:
			kdiv_list.append(calc_kernel_divergence(eigv, norm_eigvals[-1]))
	else:
		for eigv_pair in zip(norm_eigvals[:-1], norm_eigvals[1:]):
			kdiv_list.append(calc_kernel_divergence(*eigv_pair))
	return kdiv_list


def calc_all_simulations_kernel_divergence(all_simu_res, *,
		eigv_pfrac=None, eigv_pnum=None, compare_to_last=False)\
		-> numpy.ndarray:
	kdivs_res = [calc_simulation_kernel_divergence(i, eigv_pfrac=eigv_pfrac,
			eigv_pnum=eigv_pnum, compare_to_last=compare_to_last)
		for i in all_simu_res]
	return numpy.asarray(kdivs_res)


def main():
	args = get_args()
	sampling_res = load_sampling_results(args.input)
	# calculate kernel divergence
	kdivs_res = calc_all_simulations_kernel_divergence(sampling_res["results"],
		eigv_pfrac=args.eigval_preserve_frac,
		eigv_pnum=args.eigval_preserve_num,
		compare_to_last=args.compare_to_last)
	# output
	_, n_kdivs = kdivs_res.shape
	# if compare_to_last == False, there will be 1-less; in this case i throw
	# away the last
	depths = sampling_res["depth_list"][:n_kdivs]
	header = ("\t").join([str(i) for i in depths])
	numpy.savetxt(args.output, kdivs_res, fmt="%f", delimiter=args.delimiter,
		header=header)
	return


if __name__ == "__main__":
	main()
