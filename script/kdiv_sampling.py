#!/usr/bin/env python3

import argparse
import functools
import importlib
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


class PositiveInt(int):
	@functools.wraps(int.__new__)
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if new <= 0:
			raise ValueError("PositiveInt cannot be %d" % new)
		return new


class NonNegativeInt(int):
	@functools.wraps(int.__new__)
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if new < 0:
			raise ValueError("NonNegativeInt cannot be %d" % new)
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
	# default values
	dfl_delimiter = Char("\t")
	dfl_depth_min = PositiveInt(3)
	dfl_depth_step = PositiveInt(1)
	dfl_num_repeats = PositiveInt(1000)

	ap = argparse.ArgumentParser()
	ap.add_argument("input", type=str, nargs="?", default="-",
		help="input dataset to run kernel divergence sampling simulation; "
			"the input should be a plain data matrix with each row as a sample "
			"(data point), and in plain txt format (default: read from stdin)")
	ap.add_argument("--skip-lines", type=NonNegativeInt, default=0,
		metavar="int",
		help="skip first <int> lines in the input file (default: 0)")
	ap.add_argument("-d", "--delimiter", type=Char, metavar="char",
		default=dfl_delimiter,
		help="delimiter in input dataset file (default: <tab>) ")
	ap.add_argument("-o", "--output", type=str, metavar="json",
		default="-",
		help="output simulated eigenvalues at each step; this output serves "
			"as the input to th analysis script (default: stdout)")
	# parameters
	ap.add_argument("--depth-min", type=PositiveInt, metavar="int",
		default=dfl_depth_min,
		help="minimal sampling depth to run the simulation (default: %d)"
			% dfl_depth_min)
	ap.add_argument("--depth-max", type=PositiveInt, metavar="int",
		default=None,
		help="maximum sampling depth to run the simulation; by default, it "
			"equals the number of samples in input dataset")
	ap.add_argument("-k", "--batch-size", type=PositiveInt, metavar="int",
		default=dfl_depth_step,
		help="the sampling depth increment in each step a.k.a. batch size k "
			"(default: %d)" % dfl_depth_step)
	ap.add_argument("-r", "--num-repeats", type=PositiveInt, metavar="int",
		default=dfl_num_repeats,
		help="the number of simulation repeats (default: 1000)")
	ap.add_argument("--progress", action="store_true",
		help="show progressive bar (default: no)")
	# parse and refine args
	args = ap.parse_args()
	if args.input == "-":
		args.input = sys.stdin
	if args.output == "-":
		args.output = sys.stdout
	return args


def load_dataset(file, *, delimiter: str = "\t", skip_lines: int = 0):
	# file can be either fp or str, passed to numpy.loadtxt()
	return numpy.loadtxt(file, dtype=float, delimiter=delimiter,
		skiprows=skip_lines)


def progress_range(*ka, progress=False):
	"""
	range with progress bar support from module tqdm
	tqdm will be imported only if invoked with progress=True

	*ka: arguments forwarded to conventional range() call
	"""
	if progress:
		try:
			tqdm = importlib.import_module("tqdm")
			return tqdm.tqdm(range(*ka))
		except ModuleNotFoundError:
			print("option '--progress' requires tqdm module", file=sys.stderr)
			sys.exit(1)
	else:
		return range(*ka)
	return


def get_simulation_depth_list(dp_min, dp_max, dp_size):
	return list(range(dp_min, dp_max + 1, dp_size))


def centering(dist_mat, inplace=False):
	ret = dist_mat if inplace else dist_mat.copy()
	ret -= ret.mean(axis=0, keepdims=True)
	ret -= ret.mean(axis=1, keepdims=True)
	return ret


def get_rbf_kernel_gamma(data_mat) -> float:
	"""
	the 'default' gamma settings in our method; it empirically uses median of
	the pairwise euclidean distances as sigma; gamma is calculated as:

	gamma = 1 / ( 2 sigma^2 )
	"""
	# data_mat is of samples, not distance matrix
	eucl_mat = sklearn.metrics.pairwise_distances(data_mat,
		metric="euclidean")
	sigma = numpy.median(eucl_mat)
	gamma = 1.0 / (2.0 * sigma * sigma)
	return gamma


def get_rbf_kernel_mat(data_mat, gamma=None) -> numpy.ndarray:
	"""
	calculate the rbf kernel matrix from data_mat, using a precomputed gamma
	or get_rbf_kernel_gamma() by default;
	"""
	# data_mat is of samples, not distance matrix
	gamma = get_rbf_kernel_gamma(data_mat) if gamma is None else gamma
	kernel_mat = sklearn.metrics.pairwise.pairwise_kernels(data_mat,
		metric="rbf", gamma=gamma)
	return kernel_mat


def get_rbf_kpc_eigvals(data_mat, gamma=None) -> numpy.ndarray:
	"""
	calculate the rbf kernel matrix eigenvalues from data_mat; using a
	precomputed gamma or get_rbf_kernel_gamma() by default in rbf kernel
	calculations; return eigenvalues are sorted in DESCENDING order rather
	than the numpy default ascending order;
	"""
	# data_mat is of samples, not distance matrix
	kernel_mat = get_rbf_kernel_mat(data_mat, gamma=gamma)
	eigvals = numpy.linalg.eigvalsh(centering(kernel_mat))
	# now the eigvals are in asending order, by default
	eigvals = numpy.flip(eigvals)
	assert (numpy.diff(eigvals) <= 0).all()
	return eigvals


def simulation_per_round_with_permutation(data_mat, depth_list) -> (list, list):
	# permutation of the data mat
	permu_index = numpy.random.permutation(len(data_mat))
	data_mat_permu = data_mat[permu_index]
	# use the gamma calculated from the whole dataset
	gamma = get_rbf_kernel_gamma(data_mat_permu)
	eigvals_res = list()
	for depth in depth_list:
		# 'select' the first <depth> samples from this permutation;
		# emulate the 'sampling process'
		# this does not check if <depth> >= len(data_mat) (exception-safe, but
		# will ends up in 0-kernel-divergence, tho without issues
		samples = data_mat_permu[:depth]
		eigvals = get_rbf_kpc_eigvals(samples, gamma=gamma)
		eigvals_res.append(eigvals.tolist())
	return permu_index.tolist(), eigvals_res


def main():
	args = get_args()
	dataset = load_dataset(args.input, delimiter=args.delimiter,
		skip_lines=args.skip_lines)
	depth_list = get_simulation_depth_list(
		dp_min=args.depth_min,
		dp_max=(len(dataset) if args.depth_max is None else args.depth_max),
		dp_size=args.batch_size)
	# simulation
	permu_index_list = list()
	eigvals_res_list = list()
	for i in progress_range(args.num_repeats, progress=args.progress):
		permu, eigvals = simulation_per_round_with_permutation(dataset,
			depth_list)
		permu_index_list.append(permu)
		eigvals_res_list.append(eigvals)
	# output
	res = dict()
	res["depth_list"] = depth_list
	res["num_repeats"] = args.num_repeats
	res["permutations"] = permu_index_list
	res["results"] = eigvals_res_list
	with get_fp(args.output, "w") as fp:
		json.dump(res, fp, sort_keys=True)
	return


if __name__ == "__main__":
	main()
