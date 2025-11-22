use core::cell::RefCell;
use core::ops::{Deref, DerefMut};

use arrayvec::ArrayVec;
use egui_plot::PlotPoint;
use rayon::iter::{
	IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::scope;

#[repr(align(128))]
struct CachePadded<T>(T);
impl<T> Deref for CachePadded<T> {
	type Target = T;
	fn deref(&self) -> &Self::Target { &self.0 }
}
impl<T> DerefMut for CachePadded<T> {
	fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

pub struct MarchingSquaresCache {
	grids: RefCell<Vec<Vec<CachePadded<Vec<f64>>>>>,
	// polyline_builders: RefCell<Vec<PolylineBuilder>>,
}
impl Default for MarchingSquaresCache {
	fn default() -> Self {
		Self {
			grids: RefCell::new(Vec::new()),
			// polyline_builders: RefCell::new(Vec::new())
		}
	}
}
impl MarchingSquaresCache {
	fn get_grid(&self, resolution: usize) -> Vec<CachePadded<Vec<f64>>> {
		let mut grid = self.grids.borrow_mut().pop().unwrap_or_default();
		grid.resize_with(resolution + 1, || CachePadded(Vec::with_capacity(resolution + 1)));
		grid
	}
	fn return_grid(&self, grid: Vec<CachePadded<Vec<f64>>>) {
		//     println!("RETURNING GRID len {}", grid.len());
		self.grids.borrow_mut().push(grid);
	}
	// fn get_polyline_builder(&self, grid_precision: f64, eps: f64) -> PolylineBuilder {
	// 	let mut polyline_builder = self
	// 		.polyline_builders
	// 		.borrow_mut()
	// 		.pop()
	// 		.unwrap_or_else(|| PolylineBuilder::new(grid_precision, eps));
	// 	polyline_builder.eps = eps;
	// 	polyline_builder.precision_recip = 1.0 / grid_precision;
	// 	polyline_builder.endpoint_map.clear();
	// 	polyline_builder
	// }
	// fn return_polyline_builder(&self, polyline_builder: PolylineBuilder) {
	// 	self.polyline_builders.borrow_mut().push(polyline_builder);
	// }
}

pub type MarchingSquaresResult = Result<Vec<((f64, f64), Vec<Vec<PlotPoint>>)>, String>;
pub fn marching_squares<C>(
	f: impl Fn(&mut C, f64, f64) -> Result<f64, String> + Sync, bounds_min: (f64, f64),
	bounds_max: (f64, f64), resolution: usize, thread_prepare: impl Fn() -> C + Sync,
	cache: &MarchingSquaresCache,
) -> MarchingSquaresResult {
	scope!("marching_squares");
	let (x_min, y_min) = bounds_min;
	let (x_max, y_max) = bounds_max;

	let dx = (x_max - x_min) / resolution as f64;
	let dy = (y_max - y_min) / resolution as f64;

	let mut grid;
	{
		scope!("grid_calc");

		let mut error = std::sync::Mutex::new(None);
		grid = cache.get_grid(resolution);

		grid.par_iter_mut().enumerate().for_each(|(i, grid_i)| {
			scope!("grid_calc_par");
			let mut ctx = thread_prepare();

			grid_i.resize(resolution + 1, 0.0);
			let y = y_min + i as f64 * dy;
			for j in 0..=resolution {
				let x = x_min + j as f64 * dx;
				match f(&mut ctx, x, y) {
					Ok(v) => {
						grid_i[j] = v;
					},
					Err(e) => {
						*error.lock().unwrap() = Some(e);
						return;
					},
				}
			}
		});

		if let Some(error) = error.get_mut().unwrap().take() {
			cache.return_grid(grid);
			return Err(error);
		}
	}

	let eps = f32::EPSILON as f64;

	scope!("generate_polylines");

	let num_threads = rayon::current_num_threads();
	let num_chunks = (num_threads * 2).min(resolution);
	// let num_chunks = 1;

	let mut error = std::sync::Mutex::new(None);
	let chunk_results: Vec<((f64, f64), Vec<Vec<PlotPoint>>)> = (0..num_chunks)
		.into_par_iter()
		.map(|chunk_idx| {
			scope!("polyline_chunk");

			#[allow(clippy::integer_division)]
			let chunk_size = resolution / num_chunks;
			let remainder = resolution % num_chunks;

			// Distribute remainder across first chunks
			let start = chunk_idx * chunk_size + chunk_idx.min(remainder);
			let end = start + chunk_size + if chunk_idx < remainder { 1 } else { 0 };

			let mut ctx = thread_prepare();
			let mut polyline_builder = PolylineBuilder::new(0.0001, eps);

			// Cache for bottom-mid values from previous row
			let mut prev_row_top_mid: Vec<(f64, bool)> = vec![(f64::NAN, false); resolution];
			let sub_dx = dx * 0.5;
			let sub_dy = dy * 0.5;

			let y_start = y_min + start as f64 * dy;
			let y_end = y_min + end as f64 * dy;
			for i in start..end {
				let mut prev_right_mid = (f64::NAN, false);
				let y = y_min + i as f64 * dy;
				// println!("NEW ROT Y = {y}");

				for j in 0..resolution {
					let x = x_min + j as f64 * dx;

					let vals = [grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]];

					let mut config = 0u8;
					for (idx, &val) in vals.iter().enumerate() {
						if val > 0.0 {
							config |= 1 << idx;
						}
					}

					if config != 0 && config != 15 {
						// Adaptive subdivision

						// Calculate mid-edge and center points
						let (left_mid, discontinuous_left) = if prev_right_mid.0.is_nan() {
							match f(&mut ctx, x, y + sub_dy) {
								Ok(v) => (
									v,
									is_edge_discontinuous(vals[0], v, vals[3], (x, y), (x, y + dy), |x, y| {
										f(&mut ctx, x, y).ok()
									}),
								),
								Err(e) => {
									*error.lock().unwrap() = Some(e);
									return ((0.0, 0.0), vec![]);
								},
							}
						} else {
							prev_right_mid
						};

						let (bot_mid, discontinuous_bottom) = if prev_row_top_mid[j].0.is_nan() {
							match f(&mut ctx, x + sub_dx, y) {
								Ok(v) => (
									v,
									is_edge_discontinuous(vals[0], v, vals[1], (x, y), (x + dx, y), |x, y| {
										f(&mut ctx, x, y).ok()
									}),
								),
								Err(e) => {
									*error.lock().unwrap() = Some(e);
									return ((0.0, 0.0), vec![]);
								},
							}
						} else {
							prev_row_top_mid[j]
						};

						let top_mid = match f(&mut ctx, x + sub_dx, y + dy) {
							Ok(v) => v,
							Err(e) => {
								*error.lock().unwrap() = Some(e);
								return ((0.0, 0.0), vec![]);
							},
						};

						let right_mid = match f(&mut ctx, x + dx, y + sub_dy) {
							Ok(v) => v,
							Err(e) => {
								*error.lock().unwrap() = Some(e);
								return ((0.0, 0.0), vec![]);
							},
						};
						let center = match f(&mut ctx, x + sub_dx, y + sub_dy) {
							Ok(v) => v,
							Err(e) => {
								*error.lock().unwrap() = Some(e);
								return ((0.0, 0.0), vec![]);
							},
						};

						// discontinuities on outer edges
						let discontinuous_right = is_edge_discontinuous(
							vals[1],
							right_mid,
							vals[2],
							(x + dx, y),
							(x + dx, y + dy),
							|x, y| f(&mut ctx, x, y).ok(),
						);
						let discontinuous_top = is_edge_discontinuous(
							vals[2],
							top_mid,
							vals[3],
							(x + dx, y + dy),
							(x, y + dy),
							|x, y| f(&mut ctx, x, y).ok(),
						);

						// Store for next iteration

						// Process 4 subcells
						// Bottom-left subcell
						if !discontinuous_bottom && !discontinuous_left {
							process_subcell(
								x,
								y,
								sub_dx,
								sub_dy,
								[vals[0], bot_mid, center, left_mid],
								&mut polyline_builder,
								eps,
							);
						}

						// Bottom-right subcell
						if !discontinuous_bottom && !discontinuous_right {
							process_subcell(
								x + sub_dx,
								y,
								sub_dx,
								sub_dy,
								[bot_mid, vals[1], right_mid, center],
								&mut polyline_builder,
								eps,
							);
						}

						// Top-right subcell
						if !discontinuous_right && !discontinuous_top {
							process_subcell(
								x + sub_dx,
								y + sub_dy,
								sub_dx,
								sub_dy,
								[center, right_mid, vals[2], top_mid],
								&mut polyline_builder,
								eps,
							);
						}

						// Top-left subcell
						if !discontinuous_left && !discontinuous_top {
							process_subcell(
								x,
								y + sub_dy,
								sub_dx,
								sub_dy,
								[left_mid, center, top_mid, vals[3]],
								&mut polyline_builder,
								eps,
							);
						}
						prev_right_mid = (right_mid, discontinuous_right);
						prev_row_top_mid[j] = (top_mid, discontinuous_top);
					} else {
						// No subdivision needed
						// for (edge1, edge2) in get_edges_for_config(config) {
						// 	let p1 = interpolate_edge(edge1, x, y, dx, dy, &vals);
						// 	let p2 = interpolate_edge(edge2, x, y, dx, dy, &vals);

						// 	let start = PlotPoint::new(p1.0, p1.1);
						// 	let end = PlotPoint::new(p2.0, p2.1);
						// 	if !equals(start, end, eps) {
						// 		polyline_builder.add_segment(start, end);
						// 	}
						// }

						// Reset cache entries since we didn't subdivide
						prev_right_mid = (f64::NAN, false);
						prev_row_top_mid[j] = (f64::NAN, false);
					}
				}
			}

			((y_start, y_end), polyline_builder.finish())
		})
		.collect();
	if let Some(error) = error.get_mut().unwrap().take() {
		cache.return_grid(grid);
		return Err(error);
	}

	cache.return_grid(grid);
	// println!("chunk_results: {:?}", chunk_results.iter().fold(0, |acc, (_, chunk)| acc + chunk.len()));
	// let mut error = std::sync::Mutex::new(None);
	// let mut merged = merge_polyline_chunks2(chunk_results, bounds_min.1, bounds_max.1, eps);
	// merged.par_iter_mut().for_each(|polyline| {
	// 	scope!("polyline_minimize");
	// 	polyline.par_chunks_mut(32).for_each(|chunk| {
	// 		scope!("polyline_chunk_minimize");
	// 		let mut ctx = thread_prepare();
	// 		for point in chunk {
	// 			match newton_raphson_minimizer2::<1>(point.x, point.y, eps, |x, y| f(&mut ctx, x, y)) {
	// 				Ok(Some((x, y))) => {
	// 					point.x = x;
	// 					point.y = y;
	// 				},
	// 				Ok(None) => {},
	// 				Err(e) => {
	// 					*error.lock().unwrap() = Some(e);
	// 					return;
	// 				},
	// 			}
	// 		}
	// 	});
	// });
	// if let Some(error) = error.get_mut().unwrap().take() {
	// 	return Err(error);
	// }
	// println!("merged: {:?}", merged.len());
	Ok(chunk_results)
}
/// edge might contain discontinuity rather than a zero crossing
fn is_edge_discontinuous(
	val_start: f64, val_mid: f64, val_end: f64, start_param: (f64, f64), end_param: (f64, f64),
	mut eval: impl FnMut(f64, f64) -> Option<f64>,
) -> bool {
	// Only check if there's a sign change across the full edge
	if val_start.signum() == val_end.signum() {
		return false;
	}

	let abs_change = (val_end - val_start).abs();
	let left_change = (val_mid - val_start).abs();
	let right_change = (val_end - val_mid).abs();
	let max_half = left_change.max(right_change);

	const TOLERANCE: f64 = 0.95;
	if max_half < TOLERANCE * abs_change {
		return false;
	}
	// potential discontinuity
	let mid_param = ((start_param.0 + end_param.0) * 0.5, (start_param.1 + end_param.1) * 0.5);

	// Determine which half has the larger change
	if left_change > right_change {
		// Check left half: val_start -> val_mid
		let left_mid_param = ((start_param.0 + mid_param.0) * 0.5, (start_param.1 + mid_param.1) * 0.5);

		if let Some(val_left_mid) = eval(left_mid_param.0, left_mid_param.1) {
			let left_abs_change = (val_mid - val_start).abs();
			let left_left_change = (val_left_mid - val_start).abs();
			let left_right_change = (val_mid - val_left_mid).abs();
			let left_max_half = left_left_change.max(left_right_change);

			// If still highly asymmetric, it's a discontinuity
			if left_max_half > TOLERANCE * left_abs_change {
				return true;
			}
		}
	} else {
		let right_mid_param = ((mid_param.0 + end_param.0) * 0.5, (mid_param.1 + end_param.1) * 0.5);

		if let Some(val_right_mid) = eval(right_mid_param.0, right_mid_param.1) {
			let right_abs_change = (val_end - val_mid).abs();
			let right_left_change = (val_right_mid - val_mid).abs();
			let right_right_change = (val_end - val_right_mid).abs();
			let right_max_half = right_left_change.max(right_right_change);

			// If still highly asymmetric, it's a discontinuity
			if right_max_half > TOLERANCE * right_abs_change {
				return true;
			}
		}
	}

	false
}

fn process_subcell(
	x: f64, y: f64, dx: f64, dy: f64, vals: [f64; 4], polyline_builder: &mut PolylineBuilder, eps: f64,
) {
	let mut config = 0u8;
	for (idx, &val) in vals.iter().enumerate() {
		if val > 0.0 {
			config |= 1 << idx;
		}
	}

	for (edge1, edge2) in get_edges_for_config(config) {
		let p1 = interpolate_edge(edge1, x, y, dx, dy, &vals);
		let p2 = interpolate_edge(edge2, x, y, dx, dy, &vals);

		let start = PlotPoint::new(p1.0, p1.1);
		let end = PlotPoint::new(p2.0, p2.1);
		if !equals(start, end, eps) {
			polyline_builder.add_segment(start, end);
		}
	}
}

/// Get the edges that the contour crosses for a given configuration
/// Returns pairs of edges: (entry_edge, exit_edge)
fn get_edges_for_config(config: u8) -> ArrayVec<(Edge, Edge), 2> {
	let mut result = ArrayVec::new();
	match config {
		0b0000 | 0b1111 => {}, // All same sign
		0b0001 | 0b1110 => result.push((Edge::Bottom, Edge::Left)),
		0b0010 | 0b1101 => result.push((Edge::Bottom, Edge::Right)),
		0b0011 | 0b1100 => result.push((Edge::Right, Edge::Left)),
		0b0100 | 0b1011 => result.push((Edge::Right, Edge::Top)),
		0b0101 => {
			// ambiguous
			result.push((Edge::Bottom, Edge::Right));
			result.push((Edge::Top, Edge::Left));
		},
		0b0110 | 0b1001 => result.push((Edge::Bottom, Edge::Top)),
		0b0111 | 0b1000 => result.push((Edge::Top, Edge::Left)),
		0b1010 => {
			// ambiguous
			result.push((Edge::Bottom, Edge::Left));
			result.push((Edge::Right, Edge::Top));
		},
		_ => {
			unreachable!();
		},
	}
	result
}
enum Edge {
	Bottom,
	Right,
	Top,
	Left,
}

/// Interpolate position on edge where the function crosses 0
fn interpolate_edge(edge: Edge, x: f64, y: f64, dx: f64, dy: f64, vals: &[f64; 4]) -> (f64, f64) {
	match edge {
		Edge::Bottom => {
			// corners 0 and 1
			let t = vals[0] / (vals[0] - vals[1]);
			(x + t * dx, y)
		},
		Edge::Right => {
			// corners 1 and 2
			let t = vals[1] / (vals[1] - vals[2]);
			(x + dx, y + t * dy)
		},
		Edge::Top => {
			// corners 2 and 3
			let t = vals[2] / (vals[2] - vals[3]);
			(x + (1.0 - t) * dx, y + dy)
		},
		Edge::Left => {
			// corners 3 and 0
			let t = vals[3] / (vals[3] - vals[0]);
			(x, y + (1.0 - t) * dy)
		},
	}
}

fn equals(p1: PlotPoint, p2: PlotPoint, eps: f64) -> bool {
	(p1.x - p2.x).abs() < eps && (p1.y - p2.y).abs() < eps
}
fn equals1(v: f64, v2: f64, eps: f64) -> bool { (v - v2).abs() < eps }

fn merge_polyline_chunks2(
	chunks: Vec<((f64, f64), Vec<Vec<PlotPoint>>)>, min_y: f64, max_y: f64, eps: f64,
) -> Vec<Vec<PlotPoint>> {
	let height = max_y - min_y;
	let eps = height * eps;
	let mut result = vec![];
	let mut cur_index: Vec<u32> = vec![];
	let mut next_index: Vec<u32> = vec![];

	for ((chunk_start, chunk_end), polylines) in chunks {
		for mut polyline in polylines {
			let (Some(&start), Some(&end)) = (polyline.first(), polyline.last()) else {
				continue;
			};

			if equals(start, end, eps) {
				result.push(polyline);
				continue;
			}

			// check if need to merge with previous chunk
			let near_start = equals1(start.y, chunk_start, eps) || equals1(end.y, chunk_start, eps);

			if near_start {
				// maybe merge with all matching polylines from cur_index
				for &idx in cur_index.iter() {
					let candidate = &mut result[idx as usize];
					if candidate.is_empty() {
						continue;
					}
					let cand_start = candidate[0];
					let cand_end = candidate[candidate.len() - 1];

					let (pl_start, pl_end) = (polyline[0], polyline[polyline.len() - 1]);

					if equals(cand_end, pl_start, eps) {
						//  append polyline to candidate
						candidate.extend_from_slice(&polyline[1..]);
						polyline = core::mem::take(candidate);
					} else if equals(cand_end, pl_end, eps) {
						// c append reversed polyline
						candidate.extend(polyline[..polyline.len() - 1].iter().rev().copied());
						polyline = core::mem::take(candidate);
					} else if equals(cand_start, pl_end, eps) {
						// append candidate to polyline
						polyline.extend_from_slice(&candidate[1..]);
						candidate.clear();
					} else if equals(cand_start, pl_start, eps) {
						// prepend reversed polyline
						candidate.reverse();
						candidate.extend_from_slice(&polyline[1..]);
						polyline = core::mem::take(candidate);
					}
				}
			}

			if !polyline.is_empty() {
				let new_idx = result.len() as u32;
				result.push(polyline);

				// we need to check if near chunk end OR chunk start (after merging it might still be at
				// boundary)
				let new_start = result[new_idx as usize][0];
				let new_end = result[new_idx as usize].last().copied().unwrap();

				let near_chunk_end =
					equals1(new_start.y, chunk_end, eps) || equals1(new_end.y, chunk_end, eps);
				let still_near_chunk_start =
					equals1(new_start.y, chunk_start, eps) || equals1(new_end.y, chunk_start, eps);

				if near_chunk_end {
					next_index.push(new_idx);
				}
				if still_near_chunk_start {
					cur_index.push(new_idx);
				}
			}
		}

		core::mem::swap(&mut cur_index, &mut next_index);
		next_index.clear();
	}

	// empty polylines were merged
	result.retain(|pl| !pl.is_empty());

	result
}
#[derive(Clone, Copy)]
struct HashablePoint {
	x: i64,
	y: i64,
}

impl HashablePoint {
	fn from_point(p: PlotPoint, precision_recip: f64) -> Self {
		// quantize to grid
		HashablePoint { x: (p.x * precision_recip).round() as i64, y: (p.y * precision_recip).round() as i64 }
	}
}

impl PartialEq for HashablePoint {
	fn eq(&self, other: &Self) -> bool { self.x == other.x && self.y == other.y }
}

impl Eq for HashablePoint {}

impl core::hash::Hash for HashablePoint {
	fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
		self.x.hash(state);
		self.y.hash(state);
	}
}

struct PolylineBuilder {
	polylines:       Vec<Vec<PlotPoint>>,
	/// (polyline index, is_start)
	endpoint_map:    FxHashMap<HashablePoint, SmallVec<[(u32, bool); 2]>>,
	eps:             f64,
	precision_recip: f64,
}

impl PolylineBuilder {
	fn new(grid_precision: f64, eps: f64) -> Self {
		PolylineBuilder {
			polylines: Vec::new(),
			endpoint_map: FxHashMap::default(),
			eps,
			precision_recip: 1.0 / grid_precision,
		}
	}

	fn add_segment(&mut self, start: PlotPoint, end: PlotPoint) {
		let start_hash = HashablePoint::from_point(start, self.precision_recip);
		let end_hash = HashablePoint::from_point(end, self.precision_recip);

		let mut connects_at_start: Option<(usize, bool)> = None;
		let mut connects_at_end: Option<(usize, bool)> = None;

		// Check if segment.start connects to any polyline endpoint
		if let Some(candidates) = self.endpoint_map.get(&start_hash) {
			for &(idx, is_start) in candidates {
				let idx = idx as usize;
				if idx >= self.polylines.len() {
					// Skip stale indices
					continue;
				}
				let endpoint =
					if is_start { self.polylines[idx][0] } else { *self.polylines[idx].last().unwrap() };

				if equals(endpoint, start, self.eps) {
					connects_at_start = Some((idx, is_start));
					break;
				}
			}
		}

		// Check if segment.end connects to any polyline endpoint
		if let Some(candidates) = self.endpoint_map.get(&end_hash) {
			for &(idx, is_start) in candidates {
				let idx = idx as usize;
				if idx >= self.polylines.len() {
					// Skip stale indices
					continue;
				}
				let endpoint =
					if is_start { self.polylines[idx][0] } else { *self.polylines[idx].last().unwrap() };

				if equals(endpoint, end, self.eps) {
					connects_at_end = Some((idx, is_start));
					break;
				}
			}
		}

		match (connects_at_start, connects_at_end) {
			// segment connects to 2 points
			(Some((idx1, at_start1)), Some((idx2, at_start2))) => {
				if idx1 != idx2 {
					// Segment connects 2 different polylines

					self.merge_polylines(idx1, at_start1, idx2, at_start2);
				} else {
					// Segment closes a loop in 1 polyline

					// connect the start and end
					if at_start1 {
						self.polylines[idx1].push(start);
						// self.polylines[idx1].insert(0, end);
					} else {
						self.polylines[idx1].push(end);
					}
					// loop closed, no more endpoints
					self.remove_endpoint(start, idx1, at_start1);
					self.remove_endpoint(end, idx2, at_start2);
				}
			},

			// segment connects to 1 polyline at 1 point
			(Some((idx, at_start)), None) | (None, Some((idx, at_start))) => {
				// if segment connects at start, add its end point. or vice versa
				let point = if connects_at_start.is_some() { end } else { start };

				if at_start {
					// add to polyline start
					self.remove_endpoint(point, idx, true);
					self.polylines[idx].insert(0, point);
					self.add_endpoint(point, idx, true);
				} else {
					// push to polyline end
					self.remove_endpoint(start, idx, false);
					self.polylines[idx].push(point);
					self.add_endpoint(point, idx, false);
				}
			},

			// New disconnected polyline
			(None, None) => {
				let idx = self.polylines.len();
				self.polylines.push(vec![start, end]);
				self.add_endpoint(start, idx, true);
				self.add_endpoint(end, idx, false);
			},
		}
	}

	fn add_endpoint(&mut self, point: PlotPoint, polyline_idx: usize, is_start: bool) {
		let hash = HashablePoint::from_point(point, self.precision_recip);
		self.endpoint_map.entry(hash).or_default().push((polyline_idx as u32, is_start));
	}

	fn remove_endpoint(&mut self, point: PlotPoint, polyline_idx: usize, is_start: bool) {
		let hash = HashablePoint::from_point(point, self.precision_recip);
		if let Some(entries) = self.endpoint_map.get_mut(&hash) {
			entries.retain(|&mut (idx, start)| idx as usize != polyline_idx || start != is_start);
			if entries.is_empty() {
				self.endpoint_map.remove(&hash);
			}
		}
	}

	fn merge_polylines(&mut self, idx1: usize, at_start1: bool, idx2: usize, at_start2: bool) {
		let p1 = if at_start1 { self.polylines[idx1][0] } else { *self.polylines[idx1].last().unwrap() };
		let p2 = if at_start2 { self.polylines[idx2][0] } else { *self.polylines[idx2].last().unwrap() };

		self.remove_endpoint(p1, idx1, at_start1);
		self.remove_endpoint(p2, idx2, at_start2);

		let (keep_idx, remove_idx) = if idx1 < idx2 { (idx1, idx2) } else { (idx2, idx1) };

		let mut removed = self.polylines.swap_remove(remove_idx);

		// Update indices
		if remove_idx < self.polylines.len() {
			let swapped_from_idx = self.polylines.len();
			self.update_polyline_index(swapped_from_idx, remove_idx);
		}

		let mut keep = core::mem::take(&mut self.polylines[keep_idx]);

		// Determine which configuration we have:
		// - idx1 end connects to idx2 start: append idx2 to idx1
		// - idx1 start connects to idx2 end: prepend idx1 to idx2
		// - idx1 end connects to idx2 end: reverse idx2, then append
		// - idx1 start connects to idx2 start: reverse idx2, then prepend

		if keep_idx == idx1 {
			// keep idx1
			match (at_start1, at_start2) {
				(false, true) => {
					// idx1.end to idx2.start: append idx2 to idx1
					keep.extend(removed);
				},
				(false, false) => {
					// idx1.end to idx2.end: reverse idx2, then append
					removed.reverse();
					keep.extend(removed);
				},
				(true, true) => {
					// idx1.start to idx2.start: reverse idx2, then prepend
					removed.reverse();
					removed.extend(keep);
					keep = removed;
				},
				(true, false) => {
					// idx1.start to idx2.end: prepend idx2 to idx1
					removed.extend(keep);
					keep = removed;
				},
			}
		} else {
			// keep idx2
			match (at_start1, at_start2) {
				(false, true) => {
					// idx1.end to idx2.start: prepend idx1 to idx2
					removed.extend(keep);
					keep = removed;
				},
				(false, false) => {
					// idx1.end to idx2.end: reverse idx1, then append to idx2
					removed.reverse();
					keep.extend(removed);
				},
				(true, true) => {
					// idx1.start to idx2.start: reverse idx1, then prepend to idx2
					removed.reverse();
					removed.extend(keep);
					keep = removed;
				},
				(true, false) => {
					// idx1.start to idx2.end: append idx1 to idx2
					keep.extend(removed);
				},
			}
		}

		self.polylines[keep_idx] = keep;

		let new_start = self.polylines[keep_idx][0];
		let new_end = *self.polylines[keep_idx].last().unwrap();
		self.add_endpoint(new_start, keep_idx, true);
		self.add_endpoint(new_end, keep_idx, false);
	}
	fn update_polyline_index(&mut self, old_idx: usize, new_idx: usize) {
		for entries in self.endpoint_map.values_mut() {
			for entry in entries {
				if entry.0 as usize == old_idx {
					entry.0 = new_idx as u32;
				}
			}
		}
	}

	fn finish(&mut self) -> Vec<Vec<PlotPoint>> { core::mem::take(&mut self.polylines) }
}
