use core::cell::RefCell;
use core::ops::{Deref, DerefMut};

use arrayvec::ArrayVec;
use eframe::egui::{Color32, Mesh, Pos2};
use eframe::epaint::Vertex;
use egui_plot::{PlotBounds, PlotPoint};
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
#[derive(Clone, Copy)]
pub enum MarchingSquaresFill {
	Negative,
	Positive,
}
pub struct MarchingSquaresParams {
	pub resolution: usize,
	pub bounds_min: (f64, f64),
	pub bounds_max: (f64, f64),
	pub draw_lines: bool,
	pub draw_fill:  Option<MarchingSquaresFill>,
	pub fill_color: Color32,
}

#[derive(Default)]
pub struct MarchingSquaresResult {
	pub y_bounds: (f64, f64),
	pub lines:    Vec<Vec<PlotPoint>>,
	pub mesh:     MeshBuilder,
}

pub fn marching_squares<C>(
	params: MarchingSquaresParams, f: impl Fn(&mut C, f64, f64) -> Result<f64, String> + Sync,
	thread_prepare: impl Fn() -> C + Sync, cache: &MarchingSquaresCache,
) -> Result<Vec<MarchingSquaresResult>, String> {
	scope!("marching_squares");
	let (x_min, y_min) = params.bounds_min;
	let (x_max, y_max) = params.bounds_max;

	let dx = (x_max - x_min) / params.resolution as f64;
	let dy = (y_max - y_min) / params.resolution as f64;

	let mut grid;
	{
		scope!("grid_calc");

		let mut error = std::sync::Mutex::new(None);
		grid = cache.get_grid(params.resolution);

		grid.par_iter_mut().enumerate().for_each(|(i, grid_i)| {
			scope!("grid_calc_par");
			let mut ctx = thread_prepare();

			grid_i.resize(params.resolution + 1, 0.0);
			let y = y_min + i as f64 * dy;
			for j in 0..=params.resolution {
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
	let num_chunks = (num_threads * 2).min(params.resolution);
	// let num_chunks = 1;

	let mut error = std::sync::Mutex::new(None);
	let chunk_results: Vec<MarchingSquaresResult> = (0..num_chunks)
		.into_par_iter()
		.map(|chunk_idx| {
			scope!("polyline_chunk");

			#[allow(clippy::integer_division)]
			let chunk_size = params.resolution / num_chunks;
			let remainder = params.resolution % num_chunks;

			// Distribute remainder across first chunks
			let start = chunk_idx * chunk_size + chunk_idx.min(remainder);
			let end = start + chunk_size + if chunk_idx < remainder { 1 } else { 0 };

			let mut ctx = thread_prepare();
			let mut polyline_builder = PolylineBuilder::new(0.0001, eps);
			let mut mesh_builder = MeshBuilder::new(params.fill_color);

			// Cache for bottom-mid values from previous row
			let mut prev_row_top_mid: Vec<(f64, bool)> = vec![(f64::NAN, false); params.resolution];
			let sub_dx = dx * 0.5;
			let sub_dy = dy * 0.5;

			let y_start = y_min + start as f64 * dy;
			let y_end = y_min + end as f64 * dy;
			for i in start..end {
				let mut prev_right_mid = (f64::NAN, false);
				let y = y_min + i as f64 * dy;
				// println!("NEW ROT Y = {y}");
				let mut prev_was_square = false;

				for j in 0..params.resolution {
					let x = x_min + j as f64 * dx;

					let vals = [grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]];

					let config = get_config(&vals, params.draw_fill);

					// Process mesh for config 0 or 15 (full squares)
					if let Some(ref fill_type) = params.draw_fill {
						let is_positive = matches!(fill_type, MarchingSquaresFill::Positive);

						if config == 0 {
							if !is_positive {
								let can_merge = prev_was_square;
								mesh_builder.add_square(x, y, dx, dy, can_merge);
								prev_was_square = true;
							} else {
								prev_was_square = false;
							}
						} else if config == 15 {
							if is_positive {
								let can_merge = prev_was_square;
								mesh_builder.add_square(x, y, dx, dy, can_merge);
								prev_was_square = true;
							} else {
								prev_was_square = false;
							}
						} else {
							prev_was_square = false;
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
									return Default::default();
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
									return Default::default();
								},
							}
						} else {
							prev_row_top_mid[j]
						};

						let top_mid = match f(&mut ctx, x + sub_dx, y + dy) {
							Ok(v) => v,
							Err(e) => {
								*error.lock().unwrap() = Some(e);
								return Default::default();
							},
						};

						let right_mid = match f(&mut ctx, x + dx, y + sub_dy) {
							Ok(v) => v,
							Err(e) => {
								*error.lock().unwrap() = Some(e);
								return Default::default();
							},
						};
						let center = match f(&mut ctx, x + sub_dx, y + sub_dy) {
							Ok(v) => v,
							Err(e) => {
								*error.lock().unwrap() = Some(e);
								return Default::default();
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

						// Process 4 subcells
						// Bottom-left subcell
						if !discontinuous_bottom && !discontinuous_left {
							if params.draw_lines {
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
							if let Some(fill_type) = params.draw_fill {
								process_subcell_mesh(
									x,
									y,
									sub_dx,
									sub_dy,
									[vals[0], bot_mid, center, left_mid],
									&mut mesh_builder,
									fill_type,
								);
							}
						}

						// Bottom-right subcell
						if !discontinuous_bottom && !discontinuous_right {
							if params.draw_lines {
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
							if let Some(fill_type) = params.draw_fill {
								process_subcell_mesh(
									x + sub_dx,
									y,
									sub_dx,
									sub_dy,
									[bot_mid, vals[1], right_mid, center],
									&mut mesh_builder,
									fill_type,
								);
							}
						}

						// Top-right subcell
						if !discontinuous_right && !discontinuous_top {
							if params.draw_lines {
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
							if let Some(fill_type) = params.draw_fill {
								process_subcell_mesh(
									x + sub_dx,
									y + sub_dy,
									sub_dx,
									sub_dy,
									[center, right_mid, vals[2], top_mid],
									&mut mesh_builder,
									fill_type,
								);
							}
						}

						// Top-left subcell
						if !discontinuous_left && !discontinuous_top {
							if params.draw_lines {
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
							if let Some(fill_type) = params.draw_fill {
								process_subcell_mesh(
									x,
									y + sub_dy,
									sub_dx,
									sub_dy,
									[left_mid, center, top_mid, vals[3]],
									&mut mesh_builder,
									fill_type,
								);
							}
						}
						prev_right_mid = (right_mid, discontinuous_right);
						prev_row_top_mid[j] = (top_mid, discontinuous_top);
					} else {
						// No subdivision needed

						// Reset cache entries since we didn't subdivide
						prev_right_mid = (f64::NAN, false);
						prev_row_top_mid[j] = (f64::NAN, false);
					}
				}
			}

			MarchingSquaresResult {
				y_bounds: (y_start, y_end),

				lines: polyline_builder.finish(),
				mesh:  mesh_builder,
			}
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
		// println!("MORE");
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
	let config = get_config(&vals, None);

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
fn get_config(vals: &[f64; 4], fill_type: Option<MarchingSquaresFill>) -> u8 {
	let mut config = 0u8;
	for (idx, &val) in vals.iter().enumerate() {
		if val > 0.0 {
			config |= 1 << idx;
		} else if val < 0.0 {
		} else {
			// val is NaN or 0.0
			if let Some(MarchingSquaresFill::Negative) = fill_type {
				config |= 1 << idx;
			}
		}
	}
	config
}
#[allow(clippy::too_many_arguments)]
fn process_subcell_mesh(
	x: f64, y: f64, dx: f64, dy: f64, vals: [f64; 4], mesh_builder: &mut MeshBuilder,
	fill_type: MarchingSquaresFill,
) {
	let config = get_config(&vals, Some(fill_type));

	let is_positive = matches!(fill_type, MarchingSquaresFill::Positive);

	match config {
		0b0000 => {
			if !is_positive {
				mesh_builder.add_square(x, y, dx, dy, false);
			}
		},
		0b1111 => {
			if is_positive {
				mesh_builder.add_square(x, y, dx, dy, false);
			}
		},

		0b0001 | 0b1110 => {
			// bot-left corner different
			if (config == 0b0001) == is_positive {
				// bot-left, interp(bot), interp(left)
				let v1 = (x, y);
				let p_bot = interpolate_edge(Edge::Bottom, x, y, dx, dy, &vals);
				let v2 = (p_bot.0, p_bot.1);
				let p_left = interpolate_edge(Edge::Left, x, y, dx, dy, &vals);
				let v3 = (p_left.0, p_left.1);
				mesh_builder.add_triangle(v1, v2, v3);
			} else {
				// interp(bot), bot-right, top-right, top-left, interp(left)
				let p_bot = interpolate_edge(Edge::Bottom, x, y, dx, dy, &vals);
				let v1 = (p_bot.0, p_bot.1);
				let v2 = (x + dx, y);
				let v3 = (x + dx, y + dy);
				let v4 = (x, y + dy);
				let p_left = interpolate_edge(Edge::Left, x, y, dx, dy, &vals);
				let v5 = (p_left.0, p_left.1);
				mesh_builder.add_pentagon(v1, v2, v3, v4, v5);
			}
		},
		0b0010 | 0b1101 => {
			// bot-right corner different
			if (config == 0b0010) == is_positive {
				// bot-right, interp(right), interp(bot)
				let v1 = (x + dx, y);
				let p_right = interpolate_edge(Edge::Right, x, y, dx, dy, &vals);
				let v2 = (p_right.0, p_right.1);
				let p_bot = interpolate_edge(Edge::Bottom, x, y, dx, dy, &vals);
				let v3 = (p_bot.0, p_bot.1);
				mesh_builder.add_triangle(v1, v2, v3);
			} else {
				// interp(bot), bot-left, top-left, top-right, interp(right)
				let p_bot = interpolate_edge(Edge::Bottom, x, y, dx, dy, &vals);
				let v1 = (p_bot.0, p_bot.1);
				let v2 = (x, y);
				let v3 = (x, y + dy);
				let v4 = (x + dx, y + dy);
				let p_right = interpolate_edge(Edge::Right, x, y, dx, dy, &vals);
				let v5 = (p_right.0, p_right.1);
				mesh_builder.add_pentagon(v1, v2, v3, v4, v5);
			}
		},
		0b0100 | 0b1011 => {
			// top-right corner different
			if (config == 0b0100) == is_positive {
				// top-right, interp(top), interp(right)
				let v1 = (x + dx, y + dy);
				let p_top = interpolate_edge(Edge::Top, x, y, dx, dy, &vals);
				let v2 = (p_top.0, p_top.1);
				let p_right = interpolate_edge(Edge::Right, x, y, dx, dy, &vals);
				let v3 = (p_right.0, p_right.1);
				mesh_builder.add_triangle(v1, v2, v3);
			} else {
				// interp(right), bot-right, bot-left, top-left, interp(top)
				let p_right = interpolate_edge(Edge::Right, x, y, dx, dy, &vals);
				let v1 = (p_right.0, p_right.1);
				let v2 = (x + dx, y);
				let v3 = (x, y);
				let v4 = (x, y + dy);
				let p_top = interpolate_edge(Edge::Top, x, y, dx, dy, &vals);
				let v5 = (p_top.0, p_top.1);
				mesh_builder.add_pentagon(v1, v2, v3, v4, v5);
			}
		},
		0b1000 | 0b0111 => {
			// top-left corner different
			if (config == 0b1000) == is_positive {
				// top-left, interp(left), interp(top)
				let v1 = (x, y + dy);
				let p_left = interpolate_edge(Edge::Left, x, y, dx, dy, &vals);
				let v2 = (p_left.0, p_left.1);
				let p_top = interpolate_edge(Edge::Top, x, y, dx, dy, &vals);
				let v3 = (p_top.0, p_top.1);
				mesh_builder.add_triangle(v1, v2, v3);
			} else {
				// interp(left), bot-left, bot-right, top-right, interp(top)
				let p_left = interpolate_edge(Edge::Left, x, y, dx, dy, &vals);
				let v1 = (p_left.0, p_left.1);
				let v2 = (x, y);
				let v3 = (x + dx, y);
				let v4 = (x + dx, y + dy);
				let p_top = interpolate_edge(Edge::Top, x, y, dx, dy, &vals);
				let v5 = (p_top.0, p_top.1);
				mesh_builder.add_pentagon(v1, v2, v3, v4, v5);
			}
		},
		0b0011 | 0b1100 => {
			// vertical split
			let p_left = interpolate_edge(Edge::Left, x, y, dx, dy, &vals);
			let p_right = interpolate_edge(Edge::Right, x, y, dx, dy, &vals);
			if (config == 0b0011) == is_positive {
				// bot-left, bot-right, interp(right), interp(left)
				let v1 = (x, y);
				let v2 = (x + dx, y);
				let v3 = (p_right.0, p_right.1);
				let v4 = (p_left.0, p_left.1);
				mesh_builder.add_quad(v1, v2, v3, v4);
			} else {
				// interp(left), interp(right), top-right, top-left
				let v1 = (p_left.0, p_left.1);
				let v2 = (p_right.0, p_right.1);
				let v3 = (x + dx, y + dy);
				let v4 = (x, y + dy);
				mesh_builder.add_quad(v1, v2, v3, v4);
			}
		},
		0b0110 | 0b1001 => {
			// horizontal split
			let p_top = interpolate_edge(Edge::Top, x, y, dx, dy, &vals);
			let p_bot = interpolate_edge(Edge::Bottom, x, y, dx, dy, &vals);
			if (config == 0b0110) == is_positive {
				// interp(bot), bot-right, top-right, interp(top)
				let v1 = (p_bot.0, p_bot.1);
				let v2 = (x + dx, y);
				let v3 = (x + dx, y + dy);
				let v4 = (p_top.0, p_top.1);
				mesh_builder.add_quad(v1, v2, v3, v4);
			} else {
				// bot-left, interp(bot), interp(top), top-left
				let v1 = (x, y);
				let v2 = (p_bot.0, p_bot.1);
				let v3 = (p_top.0, p_top.1);
				let v4 = (x, y + dy);
				mesh_builder.add_quad(v1, v2, v3, v4);
			}
		},
		0b0101 => {
			// top-left and bot-right are positive
			let p_bot = interpolate_edge(Edge::Bottom, x, y, dx, dy, &vals);
			let p_right = interpolate_edge(Edge::Right, x, y, dx, dy, &vals);
			let p_top = interpolate_edge(Edge::Top, x, y, dx, dy, &vals);
			let p_left = interpolate_edge(Edge::Left, x, y, dx, dy, &vals);

			if is_positive {
				// top-left, interp(top), interp(right), bot-right, interp(bot), interp(left)
				let v1 = (x, y + dy);
				let v2 = (p_top.0, p_top.1);
				let v3 = (p_right.0, p_right.1);
				let v4 = (x + dx, y);
				let v5 = (p_bot.0, p_bot.1);
				let v6 = (p_left.0, p_left.1);
				mesh_builder.add_hexagon(v1, v2, v3, v4, v5, v6);
			} else {
				// bot-left, interp(left), interp(bot)
				let v1 = (x, y);
				let v2 = (p_left.0, p_left.1);
				let v3 = (p_bot.0, p_bot.1);
				mesh_builder.add_triangle(v1, v2, v3);
				// top-right, interp(right), interp(top)
				let v4 = (x + dx, y + dy);
				let v5 = (p_right.0, p_right.1);
				let v6 = (p_top.0, p_top.1);
				mesh_builder.add_triangle(v4, v5, v6);
			}
		},
		0b1010 => {
			// top-right and bot-left are positive
			let p_bot = interpolate_edge(Edge::Bottom, x, y, dx, dy, &vals);
			let p_right = interpolate_edge(Edge::Right, x, y, dx, dy, &vals);
			let p_top = interpolate_edge(Edge::Top, x, y, dx, dy, &vals);
			let p_left = interpolate_edge(Edge::Left, x, y, dx, dy, &vals);

			if is_positive {
				// bot-left, interp(bot), interp(right), top-right, interp(top), interp(left)
				let v1 = (x, y);
				let v2 = (p_bot.0, p_bot.1);
				let v3 = (p_right.0, p_right.1);
				let v4 = (x + dx, y + dy);
				let v5 = (p_top.0, p_top.1);
				let v6 = (p_left.0, p_left.1);
				mesh_builder.add_hexagon(v1, v2, v3, v4, v5, v6);
			} else {
				// bot-right, interp(bot), interp(right)
				let v1 = (x + dx, y);
				let v2 = (p_bot.0, p_bot.1);
				let v3 = (p_right.0, p_right.1);
				mesh_builder.add_triangle(v1, v2, v3);
				// top-left, interp(top), interp(left)
				let v4 = (x, y + dy);
				let v5 = (p_top.0, p_top.1);
				let v6 = (p_left.0, p_left.1);
				mesh_builder.add_triangle(v4, v5, v6);
			}
		},
		_ => {
			unreachable!();
		},
	}
}

pub struct MeshBuilder {
	pub color:    Color32,
	pub bounds:   PlotBounds,
	pub vertices: Vec<Vertex>,
	pub indices:  Vec<u32>,
}

impl Default for MeshBuilder {
	fn default() -> Self {
		Self {
			bounds:   PlotBounds::NOTHING,
			vertices: Default::default(),
			color:    Color32::WHITE,
			indices:  Default::default(),
		}
	}
}

impl MeshBuilder {
	pub fn is_empty(&self) -> bool { self.vertices.is_empty() }
	fn new(color: Color32) -> Self {
		Self { color, bounds: PlotBounds::NOTHING, vertices: Vec::new(), indices: Vec::new() }
	}
	fn add_vert(&mut self, v: (f64, f64)) -> Vertex {
		let (x, y) = v;
		self.bounds.extend_with(&PlotPoint::new(x, y));
		Vertex { pos: Pos2::new(x as f32, y as f32), uv: Pos2::new(0.0, 0.0), color: self.color }
	}
	fn add_triangle(&mut self, v1: (f64, f64), v2: (f64, f64), v3: (f64, f64)) {
		let base = self.vertices.len() as u32;
		let verts = [self.add_vert(v1), self.add_vert(v2), self.add_vert(v3)];
		self.vertices.extend(verts);
		self.indices.extend([base, base + 1, base + 2]);
	}

  #[rustfmt::skip]
	fn add_quad(&mut self, v1: (f64, f64), v2: (f64, f64), v3: (f64, f64), v4: (f64, f64)) {
		let base = self.vertices.len() as u32;
		let verts = [self.add_vert(v1), self.add_vert(v2), self.add_vert(v3), self.add_vert(v4)];
		self.vertices.extend(verts);
		self.indices.extend([
			base, base + 1, base + 2,
			base, base + 2, base + 3
    ]);
	}

  #[rustfmt::skip]
	fn add_pentagon(
		&mut self, v1: (f64, f64), v2: (f64, f64), v3: (f64, f64), v4: (f64, f64), v5: (f64, f64),
	) {
		let base = self.vertices.len() as u32;
		let verts =
			[self.add_vert(v1), self.add_vert(v2), self.add_vert(v3), self.add_vert(v4), self.add_vert(v5)];
		self.vertices.extend(verts);
		self.indices.extend([
 			base, base + 1, base + 2,
 			base, base + 2, base + 3,
 			base, base + 3, base + 4
		]);
	}

  #[rustfmt::skip]
	fn add_hexagon(
		&mut self, v1: (f64, f64), v2: (f64, f64), v3: (f64, f64), v4: (f64, f64), v5: (f64, f64),
		v6: (f64, f64),
	) {
		let base = self.vertices.len() as u32;
		let verts = [
			self.add_vert(v1), self.add_vert(v2), self.add_vert(v3),
			self.add_vert(v4), self.add_vert(v5), self.add_vert(v6),
		];
		self.vertices.extend(verts);
		self.indices.extend([
			base, base + 1, base + 2,
			base, base + 2, base + 3,
			base, base + 3, base + 4,
			base, base + 4, base + 5,
		]);
	}

	fn add_square(&mut self, x: f64, y: f64, dx: f64, dy: f64, can_merge: bool) {
		// bot-left, bot-right, top-right, top-left
		if can_merge {
			let new_bot_right = self.add_vert((x + dx, y));
			let new_top_right = self.add_vert((x + dx, y + dy));
			let len = self.vertices.len();
			self.vertices[len - 3] = new_bot_right;
			self.vertices[len - 2] = new_top_right;
		} else {
			let v1 = (x, y);
			let v2 = (x + dx, y);
			let v3 = (x + dx, y + dy);
			let v4 = (x, y + dy);
			self.add_quad(v1, v2, v3, v4);
		}
	}
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
