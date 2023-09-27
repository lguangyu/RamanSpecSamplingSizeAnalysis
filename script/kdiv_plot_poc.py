#!/usr/bin/env python3

import argparse
import functools
import itertools
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot
import numpy
import sys


class Char(str):
	@functools.wraps(str.__new__)
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if len(new) != 1:
			raise ValueError("Char must be a string of length 1, got %d"
				% len(new))
		return new


class PositiveFraction(float):
	@functools.wraps(float.__new__)
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if (new <= 0) or (new > 1.0):
			raise ValueError("PositiveFraction must be in (0, 1], got %f" % new)
		return new


class PositiveInt(int):
	@functools.wraps(int.__new__)
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if new <= 0:
			raise ValueError("PositiveInt cannot be %d" % new)
		return new


class CommaSepList(list):
	@classmethod
	def from_str(cls, s: str):
		return cls(s.split(","))


POC_KDIV_FILTER = dict()


def _kdiv_filter_standard(kdiv_raw: numpy.ndarray) -> numpy.ndarray:
	# use raw kdiv as per-step kdiv
	return kdiv_raw


def _kdiv_filter_conservative(kdiv_raw: numpy.ndarray) -> numpy.ndarray:
	# use the maximum kdiv appeared at this and later steps as per-step kdiv
	# this represents the "worst-case" senario observed in the sampling trials
	ret = kdiv_raw.copy()
	for col in range(kdiv_raw.shape[1]):
		ret[:, col] = kdiv_raw[:, col:].max(axis=1)
	return ret


POC_KDIV_FILTER["standard"] = _kdiv_filter_standard
POC_KDIV_FILTER["conservative"] = _kdiv_filter_conservative


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type=str, nargs="?", default="-",
		help="input analysis outputs from kdiv_analysis.py "
			"(default: read from stdin)")
	ap.add_argument("-d", "--delimiter", type=Char, metavar="char",
		default=Char("\t"),
		help="deliminter used in the input table (default: <tab>)")
	ap.add_argument("-p", "--plot", type=str, metavar="png",
		default="-",
		help="output plot image file (default: stdout)")
	ap.add_argument("--dpi", type=PositiveInt, default=PositiveInt(300),
		metavar="int",
		help="dpi of the output image (default: 300)")
	ap.add_argument("--title", type=str,
		metavar="str",
		help="add <str> as title above the plot (default: no)")
	# parameters
	ap.add_argument("-m", "--poc-filter-meth", type=str, default="standard",
		choices=sorted(POC_KDIV_FILTER.keys()),
		help="the method to filter kernel divergence for POC estimation; "
			"'standard' uses the raw kernel divergence values read from input, "
			"and 'conservative' uses the maximum observed value at each and "
			"later steps; the 'conservative' method represents the worst-case "
			"senario observed in the sampling trials (default: standard)")
	ap.add_argument("-t", "--threshold", type=CommaSepList.from_str,
		required=True, metavar="float[,float[,...]]",
		help="a comma-separated list of kernel divergence thresholds to "
			"calculate POC, for example '0.01,0.005'; values must be positive "
			"decimals within 0-1 (required)")

	# parse and refine args
	args = ap.parse_args()
	if args.input == "-":
		args.input = sys.stdin
	if args.plot == "-":
		args.plot = sys.stdout.buffer
	args.threshold = sorted([PositiveFraction(i) for i in args.threshold])
	return args


def load_kdiv_for_poc(file, *, deliminter="\t", poc_filter_meth="standard")\
		-> (numpy.ndarray, numpy.ndarray):
	raw = numpy.loadtxt(file, delimiter=deliminter, dtype=object, comments=None)
	# parse depth in the first row
	depth = raw[0]
	depth[0] = depth[0].strip("#\t")
	depth = depth.astype(int)
	# parse the rest as kdiv
	poc_filter = POC_KDIV_FILTER[poc_filter_meth]
	kdiv = poc_filter(raw[1:].astype(float))
	return depth, kdiv


def create_layout() -> dict:
	figure = matplotlib.pyplot.gcf()
	axes = matplotlib.pyplot.gca()

	# update axes style
	for sp in axes.spines.values():
		sp.set_visible(True)
	axes.tick_params(
		left=True, labelleft=True,
		right=False, labelright=False,
		top=False, labeltop=False,
		bottom=True, labelbottom=True,
	)

	return dict(figure=figure, axes=axes)


def add_poc(axes: matplotlib.axes.Axes, *, depth, kdiv_mean, threshold, color)\
		-> matplotlib.collections.PathCollection:
	# find the last poc above threshold
	above_mask = kdiv_mean > threshold
	if above_mask.all():
		# poc is larger than range
		x = list()
		y = list()
		label = "POC$_{%s}$=N/A" % str(threshold)
	elif not above_mask.any():
		# poc is smaller than range
		x = list()
		y = list()
		label = "POC$_{%s}$<%u" % (str(threshold), depth[0])
	else:
		last_above_idx = len(kdiv_mean) - numpy.argmax(above_mask[::-1])
		x = depth[last_above_idx]
		y = kdiv_mean[last_above_idx]
		label = "POC$_{%s}$=%u" % (str(threshold), x)
		# add threshold lines
		axes.axhline(threshold, linestyle="--", linewidth=1.0, color=color,
			zorder=2)
		axes.axvline(x, linestyle="--", linewidth=1.0, color=color,
			zorder=2)
	return axes.scatter(x, y, s=30, marker="o", edgecolors=color,
		facecolors="#ffffff80", zorder=3, label=label)


def plot_poc(png, depth: numpy.ndarray, kdiv: numpy.ndarray, thres: list, *,
		title=None, dpi=300):
	layout = create_layout()
	figure: matplotlib.figure.Figure = layout["figure"]
	figure.set_size_inches(4, 3)

	kdiv_mean = kdiv.mean(axis=0)
	kdiv_std = kdiv.std(axis=0)

	axes: matplotlib.axes.Axes = layout["axes"]
	color = "#5f3d8a"
	# plot mean line and std as shadow
	axes.plot(depth, kdiv_mean, linestyle="-", linewidth=1.5, color=color,
		zorder=2)
	axes.fill_between(depth, kdiv_mean - kdiv_std, kdiv_mean + kdiv_std,
		edgecolor="none", facecolor=color + "40", zorder=2)

	# add pocs
	colors = [matplotlib.colors.to_hex(i)
		for i in matplotlib.cm.get_cmap("Set1").colors]
	poc_handles = list()
	for t, color in zip(thres, itertools.cycle(colors)):
		p = add_poc(axes, depth=depth, kdiv_mean=kdiv_mean, threshold=t,
			color=color)
		poc_handles.append(p)
	axes.legend(handles=poc_handles, loc=1, bbox_to_anchor=[1.02, 1.02],
		fancybox=False, frameon=False, handlelength=0.75)

	# misc
	axes.set_xlim(0, max(depth))
	axes.set_ylim(0, max(thres) * 1.5)
	axes.set_xlabel("Sampling depth", fontsize=12)
	axes.set_ylabel("Kernel divergence", fontsize=12)
	axes.set_title(title, fontsize=12)

	# savefig and clean up
	figure.tight_layout()
	figure.savefig(png, dpi=dpi)
	matplotlib.pyplot.close()
	return


def main():
	args = get_args()
	depth, kdiv = load_kdiv_for_poc(args.input, deliminter=args.delimiter,
		poc_filter_meth=args.poc_filter_meth)
	plot_poc(args.plot, depth=depth, kdiv=kdiv, thres=args.threshold,
		title=args.title, dpi=args.dpi)
	return


if __name__ == "__main__":
	main()
