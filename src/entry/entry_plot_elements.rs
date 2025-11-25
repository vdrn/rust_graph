use alloc::sync::Arc;
use core::cell::{RefCell, RefMut};
use core::mem;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::marker::ConstParamTy;

use eframe::egui::{self, Color32, Id, Pos2, RichText, Stroke, pos2, remap};
use egui_plot::{Line, PlotItemBase, PlotPoint, PlotPoints, PlotTransform, Points, Polygon, Text};
use evalexpr::{EvalexprError, EvalexprFloat, ExpressionFunction, FlatNode, HashMapContext, Stack, Value};
use thread_local::ThreadLocal;

use crate::draw_buffer::{
	DrawLine, DrawMesh, DrawMeshType, DrawPoint, DrawPolygonGroup, DrawText, EguiPlotMesh, FillMesh, OtherPointType, PointInteraction, PointInteractionType
};
use crate::entry::{Entry, EntryType, EquationType, f64_to_value};
use crate::marching_squares::MeshBuilder;
use crate::math::{DiscontinuityDetector, pseudoangle, zoom_in_x_on_nan_boundary};
use crate::{ThreadLocalContext, marching_squares, thread_local_get};

pub struct PlotParams {
	pub eps:                 f64,
	pub first_x:             f64,
	pub last_x:              f64,
	pub first_y:             f64,
	pub last_y:              f64,
	pub step_size:           f64,
	pub step_size_y:         f64,
	pub resolution:          usize,
	pub prev_plot_transform: Option<egui_plot::PlotTransform>,
	pub invert_axes:         [bool; 2],
}
#[allow(clippy::too_many_arguments)]
#[allow(clippy::panic_in_result_fn)]
pub fn entry_create_plot_elements<T: EvalexprFloat>(
	entry: &mut Entry<T>, id: Id, sorting_idx: u32, selected_id: Option<Id>,
	ctx: &evalexpr::HashMapContext<T>, plot_params: &PlotParams,
	draw_buffer: &ThreadLocal<crate::DrawBufferRC>, tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>,
) -> Result<(), Vec<(u64, String)>> {
	// thread_local_get(tl_context).stack_overflow_guard.set(0);

	let visible = entry.active;
	if !visible && !matches!(entry.ty, EntryType::Folder { .. }) {
		return Ok(());
	}
	let color = entry.color();
	// println!("step_size: {deriv_step_x}");

	let draw_buffer_c = thread_local_get(draw_buffer);
	match &mut entry.ty {
		EntryType::Folder { entries } => {
			let mut errors = std::sync::Mutex::new(Vec::new());
			entries.par_iter_mut().enumerate().for_each(|(ei, entry)| {
				let eid = Id::new(entry.id);
				if let Err(e) = entry_create_plot_elements(
					entry,
					eid,
					sorting_idx + ei as u32,
					selected_id,
					ctx,
					plot_params,
					draw_buffer,
					tl_context,
				) {
					errors.lock().unwrap().extend(e);
				}
			});
			if errors.get_mut().unwrap().is_empty() {
				return Ok(());
			}
			return Err(errors.into_inner().unwrap());
		},
		EntryType::Constant { .. } => {},
		EntryType::Label { x, y, size, underline, .. } => {
			let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
			let mut stack = thread_local_get(tl_context).stack.borrow_mut();

			match eval_point(&mut stack, ctx, x.node.as_ref(), y.node.as_ref()) {
				Ok(Some((x, y))) => {
					let size = if let Some(size) = &size.node {
						match size.eval_float_with_context(&mut stack, ctx) {
							Ok(size) => size.to_f64() as f32,
							Err(e) => {
								return Err(vec![(entry.id, e.to_string())]);
							},
						}
					} else {
						12.0
					};
					let mut label_text = RichText::new(entry.name.clone()).size(size);
					if *underline {
						label_text = label_text.underline()
					}

					let text = Text::new(entry.name.clone(), PlotPoint { x, y }, label_text).color(color);

					draw_buffer.texts.push(DrawText::new(sorting_idx, text));
				},
				Err(e) => {
					return Err(vec![(entry.id, e)]);
				},
				_ => {},
			}
		},
		EntryType::Points { points, style } => {
			let mut stack = thread_local_get(tl_context).stack.borrow_mut();
			let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
			// main_context
			// 	.write()
			// 	.unwrap()
			// 	.set_value("x", evalexpr::Value::<T>::Float(T::ZERO))
			// 	.unwrap();
			let mut arrow_buffer = vec![];
			let mut line_buffer = vec![];
			let color_rgba = color.to_array();
			let color_outer =
				Color32::from_rgba_unmultiplied(color_rgba[0], color_rgba[1], color_rgba[1], 128);
			let mut prev_point: Option<egui::Vec2> = None;
			let arrow_scale = egui::Vec2::new(
				(plot_params.last_x - plot_params.first_x) as f32,
				(plot_params.last_y - plot_params.first_y) as f32,
			) * 0.002;
			let points_len = points.len();
			for (i, p) in points.iter_mut().enumerate() {
				match eval_point(&mut stack, ctx, p.x.node.as_ref(), p.y.node.as_ref()) {
					Ok(Some((x, y))) => {
						p.val = Some((T::from_f64(x), T::from_f64(y)));
						let point_id = id.with(i);
						let selected = selected_id == Some(id);
						let radius = if selected { 6.5 } else { 4.5 };
						let radius_outer = if selected { 12.5 } else { 7.5 };

						if style.show_points || p.drag_point.is_some() {
							draw_buffer.points.push(DrawPoint::new(
								sorting_idx,
								i as u32,
								PointInteraction {
									x,
									y,
									radius,
									ty: PointInteractionType::Other(OtherPointType::Point),
								},
								Points::new(entry.name.clone(), [x, y]).color(color).radius(radius),
							));
							if p.drag_point.is_some() {
								let selectable_point = PointInteraction {
									ty: PointInteractionType::Draggable { i: (id, i as u32) },
									x,
									y,
									radius: radius_outer,
								};
								draw_buffer.points.push(DrawPoint::new(
									sorting_idx,
									i as u32,
									selectable_point,
									Points::new(entry.name.clone(), [x, y])
										.id(point_id)
										.color(color_outer)
										.radius(radius_outer),
								));
							}
						}
						if let Some(label_config) = &style.label_config
							&& i == points_len - 1
							&& !entry.name.trim().is_empty()
						{
							let size = label_config.size.size();
							let mut label = RichText::new(entry.name.clone()).size(size);
							if label_config.italic {
								label = label.italics();
							}

							let dir = label_config.pos.dir();
							let text = Text::new(
								entry.name.clone(),
								PlotPoint {
									x: x + (dir.x * size * arrow_scale.x) as f64,
									y: y + (dir.y * size * arrow_scale.y) as f64,
								},
								label,
							)
							.color(color);

							draw_buffer.texts.push(DrawText::new(sorting_idx, text));
						}

						if style.show_lines {
							line_buffer.push([x, y]);
							let cur_point = egui::Vec2::new(x as f32, y as f32);
							if style.show_arrows
								&& let Some(pp) = prev_point
							{
								let dir = (pp - cur_point).normalized();
								let arrow_len = arrow_scale * radius_outer;
								let base = cur_point + dir * arrow_len.length();
								let a = base + dir.rot90() * arrow_len * 0.5;
								let b = base - dir.rot90() * arrow_len * 0.5;

								arrow_buffer.push(
									Polygon::new(
										"",
										vec![[x, y], [a.x as f64, a.y as f64], [b.x as f64, b.y as f64]],
									)
									.fill_color(color)
									.allow_hover(false)
									.stroke(Stroke::new(0.0, color)),
								);
							}
							prev_point = Some(cur_point);
						}
					},
					Err(e) => {
						return Err(vec![(entry.id, e)]);
					},
					_ => {},
				}
			}

			// let width = if selected { 3.5 } else { 1.0 };
			let width = 1.0;

			if line_buffer.len() > 1 && style.show_lines {
				let line = Line::new(entry.name.clone(), line_buffer)
					.color(color)
					.id(id)
					.width(style.line_style.line_width)
					.style(style.line_style.egui_line_style());
				draw_buffer.lines.push(DrawLine::new(sorting_idx, id, width, line));
				if !arrow_buffer.is_empty() {
					draw_buffer.polygons.push(DrawPolygonGroup::new(sorting_idx, arrow_buffer));
				}
				// plot_ui.line(line);
			}
		},
		EntryType::Function {
			func,
			can_be_drawn,
			identifier,
			parametric,
			parametric_fill,
			range_start,
			range_end,
			style,
			implicit_resolution,
			selectable,
			..
		} => {
			if !*can_be_drawn {
				return Ok(());
			}
			let stack_len = thread_local_get(tl_context).stack.borrow().len();
			assert!(stack_len == 0, "stack is not empty: {stack_len}");

			let display_name = if identifier.is_empty() {
				format!("f({}): {}", func.args_to_string(), func.text.trim())
			} else {
				format!("{}({}): {}", identifier, func.args_to_string(), func.text.trim())
			};

			let selected = selected_id == Some(id);
			let width = if selected { style.line_width + 2.5 } else { style.line_width };

			let mut add_line = |line: Vec<PlotPoint>| {
				if line.is_empty() {
					return;
				}
				let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
				draw_buffer.lines.push(DrawLine::new(
					sorting_idx,
					id,
					width,
					Line::new(&display_name, PlotPoints::Owned(line))
						.id(id)
						.allow_hover(*selectable)
						.width(width)
						.style(style.egui_line_style())
						.color(color),
				));
			};
			let mut add_egui_plot_mesh = |mesh: MeshBuilder| {
				if mesh.is_empty() {
					return;
				}
				let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
				draw_buffer.meshes.push(DrawMesh {
					sorting_index: sorting_idx,
					ty:            DrawMeshType::EguiPlotMesh(EguiPlotMesh {
						bounds: mesh.bounds,
						mesh: RefCell::new(mesh),
						color,
						plot_item_base: PlotItemBase::new(display_name.clone()),
					}),
				});
			};
			let fill_color = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 128);

			if let Some(expr_func) = &func.expr_function {
				if func.args.is_empty() {
					let mut stack = thread_local_get(tl_context).stack.borrow_mut();
					// horizontal line
					let value = match expr_func.call(&mut stack, ctx, &[]) {
						Ok(v) => match v {
							Value::Float(v) => v,
							Value::Float2(_, _) | Value::Boolean(_) | Value::Tuple(_) | Value::Empty => {
								return Ok(());
							},
						},
						Err(e) => return Err(vec![(entry.id, e.to_string())]),
					};
					add_line(vec![
						PlotPoint::new(plot_params.first_x, value.to_f64()),
						PlotPoint::new(plot_params.last_x, value.to_f64()),
					]);
				} else if func.args.len() == 1 {
					// starndard X or Y function
					match func.equation_type {
						EquationType::None => {
							if *parametric {
								let mut fill_mesh = if *parametric_fill
									&& let Some(plot_trans) = plot_params.prev_plot_transform
								{
									Some((FillMesh::new(fill_color), plot_trans))
								} else {
									None
								};

								let mut add_parametric_line = |line: Vec<PlotPoint>, break_fill_mesh: bool| {
									if line.is_empty() && !break_fill_mesh {
										return;
									}
									if let Some((fm, plot_trans)) = &mut fill_mesh {
										for point in line.iter() {
											let screen_point = gpu_position_from_point(
												plot_trans, plot_params.invert_axes, point,
											);
											fm.add_vertex(screen_point.x, screen_point.y);
										}

										if break_fill_mesh {
											let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
											draw_buffer.meshes.push(DrawMesh {
												sorting_index: sorting_idx,
												ty:            DrawMeshType::FillMesh(mem::take(fm)),
											});
											*fm = FillMesh::new(fill_color);
										}
									}

									add_line(line);
								};

								if func.args[0].to_str() == "y" {
									draw_parametric_function::<T, { SimpleFunctionType::Y }>(
										tl_context,
										ctx,
										plot_params,
										entry.id,
										expr_func,
										range_start.node.as_ref(),
										range_end.node.as_ref(),
										&mut add_parametric_line,
									)?;
								} else {
									draw_parametric_function::<T, { SimpleFunctionType::X }>(
										tl_context,
										ctx,
										plot_params,
										entry.id,
										expr_func,
										range_start.node.as_ref(),
										range_end.node.as_ref(),
										&mut add_parametric_line,
									)?;
								}

								if let Some((fm, _)) = fill_mesh {
									let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
									draw_buffer.meshes.push(DrawMesh {
										sorting_index: sorting_idx,
										ty:            DrawMeshType::FillMesh(fm),
									});
								}
							} else {
								if func.args[0].to_str() == "y" {
									draw_simple_function::<T, { SimpleFunctionType::Y }>(
										tl_context, ctx, plot_params, entry.id, expr_func, &mut add_line,
									)?;
								} else {
									draw_simple_function::<T, { SimpleFunctionType::X }>(
										tl_context, ctx, plot_params, entry.id, expr_func, &mut add_line,
									)?;
								}
							}
						},
						EquationType::Equality => {
							let mut stack = Stack::<T>::with_capacity(0);
							let value = match expr_func.call(&mut stack, ctx, &[Value::Float(T::ZERO)]) {
								Ok(v) => match v {
									Value::Float(v) => v,
									Value::Float2(_, _)
									| Value::Boolean(_)
									| Value::Tuple(_)
									| Value::Empty => {
										return Ok(());
									},
								},
								Err(e) => return Err(vec![(entry.id, e.to_string())]),
							};

							if func.args[0].to_str() == "x" {
								add_line(vec![
									PlotPoint::new(-value.to_f64(), plot_params.first_y),
									PlotPoint::new(-value.to_f64(), plot_params.last_y),
								]);
							} else {
								add_line(vec![
									PlotPoint::new(plot_params.first_x, -value.to_f64()),
									PlotPoint::new(plot_params.last_x, -value.to_f64()),
								]);
							}
						},
						equation_type => {
							if func.args[0].to_str() == "y" {
								draw_implicit(
									tl_context,
									plot_params,
									entry.id,
									*implicit_resolution,
									equation_type,
									fill_color,
									|cc, _, y| expr_func.call(cc, ctx, &[f64_to_value::<T>(y)]),
									&mut add_line,
									&mut add_egui_plot_mesh,
								)?;
							} else {
								draw_implicit(
									tl_context,
									plot_params,
									entry.id,
									*implicit_resolution,
									equation_type,
									fill_color,
									|cc, x, _| expr_func.call(cc, ctx, &[f64_to_value::<T>(x)]),
									&mut add_line,
									&mut add_egui_plot_mesh,
								)?;
							}
						},
					}
				} else if func.args.len() == 2 {
					if func.args[0].to_str() == "x" && func.args[1].to_str() == "y" {
						if func.equation_type != EquationType::None {
							draw_implicit(
								tl_context,
								plot_params,
								entry.id,
								*implicit_resolution,
								func.equation_type,
								fill_color,
								|cc, x, y| {
									expr_func.call(cc, ctx, &[f64_to_value::<T>(x), f64_to_value::<T>(y)])
								},
								&mut add_line,
								&mut add_egui_plot_mesh,
							)?;
						}
					}
				}
			}
		},
	}
	Ok(())
}

fn eval_point<T: EvalexprFloat>(
	stack: &mut Stack<T>, ctx: &evalexpr::HashMapContext<T>, px: Option<&FlatNode<T>>,
	py: Option<&FlatNode<T>>,
) -> Result<Option<(f64, f64)>, String> {
	let (Some(x), Some(y)) = (px, py) else {
		return Ok(None);
	};
	let x = x.eval_float_with_context(stack, ctx).map_err(|e| e.to_string())?;
	let y = y.eval_float_with_context(stack, ctx).map_err(|e| e.to_string())?;
	Ok(Some((x.to_f64(), y.to_f64())))
}

#[derive(PartialEq, Eq, ConstParamTy)]
enum SimpleFunctionType {
	X,
	Y,
}
fn draw_simple_function<T: EvalexprFloat, const TY: SimpleFunctionType>(
	tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>, ctx: &HashMapContext<T>, plot_params: &PlotParams,
	id: u64, func: &ExpressionFunction<T>, mut add_line: impl FnMut(Vec<PlotPoint>),
) -> Result<(), Vec<(u64, String)>> {
	let mut stack = thread_local_get(tl_context).stack.borrow_mut();
	let mut pp_buffer = vec![];
	let mut prev_sampling_point: Option<(f64, f64)> = None;

	let mut sampling_arg = match TY {
		SimpleFunctionType::X => plot_params.first_x,
		SimpleFunctionType::Y => plot_params.first_y,
	};
	let last_sampling_arg = match TY {
		SimpleFunctionType::X => plot_params.last_x,
		SimpleFunctionType::Y => plot_params.last_y,
	};
	let step_size = match TY {
		SimpleFunctionType::X => plot_params.step_size,
		SimpleFunctionType::Y => plot_params.step_size_y,
	};

	if let Some(constant) = func.as_constant() {
		let value = match constant {
			Value::Float(v) => v,
			Value::Float2(_, _) | Value::Boolean(_) | Value::Tuple(_) | Value::Empty => {
				return Ok(());
			},
		};
		match TY {
			SimpleFunctionType::X => {
				pp_buffer.push(PlotPoint::new(plot_params.first_x, value.to_f64()));
				pp_buffer.push(PlotPoint::new(plot_params.last_x, value.to_f64()));
			},
			SimpleFunctionType::Y => {
				pp_buffer.push(PlotPoint::new(value.to_f64(), plot_params.first_y));
				pp_buffer.push(PlotPoint::new(value.to_f64(), plot_params.last_y));
			},
		}
	} else {
		// let mut prev_y = None;
		let mut discontinuity_detector = DiscontinuityDetector::new(step_size, plot_params.eps);

		// let graph_size = match TY {
		// 	SimpleFunctionType::X => plot_params.last_y - plot_params.first_y,
		// 	SimpleFunctionType::Y => plot_params.last_x - plot_params.first_x,
		// };
		// let mut prev_angle: Option<f64> = None;
		// let mut prev_point: Option<PlotPoint> = None;
		while sampling_arg <= last_sampling_arg {
			match func.call(&mut stack, ctx, &[f64_to_value::<T>(sampling_arg)]) {
				Ok(Value::Float(val)) => {
					let val = val.to_f64();

					let zoomed_in_on_nan_boundary = prev_sampling_point.and_then(|(prev_arg, prev_val)| {
						zoom_in_x_on_nan_boundary(
							(prev_arg, prev_val),
							(sampling_arg, val),
							plot_params.eps,
							|arg| {
								func.call(&mut stack, ctx, &[f64_to_value::<T>(arg)])
									.and_then(|v| v.as_float())
									.map(|v| v.to_f64())
									.ok()
							},
						)
					});

					let mut on_nan_boundary = false;

					let (cur_arg, cur_val) = if let Some(zoomed_in_on_nan_boundary) = zoomed_in_on_nan_boundary
					{
						on_nan_boundary = true;
						zoomed_in_on_nan_boundary
					} else {
						(sampling_arg, val)
					};

					let latest_point = if !on_nan_boundary
						&& let Some((left, right)) = discontinuity_detector.detect(cur_arg, cur_val, |arg| {
							func.call(&mut stack, ctx, &[f64_to_value::<T>(arg)])
								.and_then(|v| v.as_float())
								.map(|v| v.to_f64())
								.ok()
						}) {
						match TY {
							SimpleFunctionType::X => {
								pp_buffer.push(PlotPoint::new(left.0, left.1));
								add_line(mem::take(&mut pp_buffer));
								Some(PlotPoint::new(right.0, right.1))
							},
							SimpleFunctionType::Y => {
								pp_buffer.push(PlotPoint::new(left.1, left.0));
								add_line(mem::take(&mut pp_buffer));
								Some(PlotPoint::new(right.1, right.0))
							},
						}
					} else {
						if !cur_val.is_nan() {
							Some(match TY {
								SimpleFunctionType::X => PlotPoint::new(cur_arg, cur_val),
								SimpleFunctionType::Y => PlotPoint::new(cur_val, cur_arg),
							})
						} else {
							None
						}
					};
					if let Some(latest_point) = latest_point {
						// TODO:This is commented out because we cannot just skip the points,
						// as the egui_plot hovering functionality does not work without dense
						// poitns on lines.
						// We need to implement our hovering for this optimization.
						//
						// if let Some(prev_point) = prev_point {
						// 	let dx = latest_point.x - prev_point.x;
						// 	let dy = latest_point.y - prev_point.y;
						// 	let pangle = pseudoangle(dx, dy);
						// 	if let Some(prev_angle) = prev_angle
						// 		&& pp_buffer.len() > 1
						// 	{
						// 		let angle_diff = (pangle - prev_angle).abs();
						// 		if angle_diff < 0.00001 * graph_size {
						// 			*pp_buffer.last_mut().unwrap() = latest_point;
						// 			// pp_buffer.push(latest_point);
						// 		} else {
						// 			pp_buffer.push(latest_point);
						// 		}
						// 	} else {
						// 		pp_buffer.push(latest_point);
						// 	}
						// 	prev_angle = Some(pangle);
						// }
						// prev_point = Some(latest_point);
						pp_buffer.push(latest_point);
					}

					prev_sampling_point = Some((sampling_arg, val));
				},
				Ok(Value::Empty) => {
					add_line(mem::take(&mut pp_buffer));
				},
				Ok(Value::Tuple(_) | Value::Float2(_, _)) => {
					return Err(vec![(
						id,
						"Non-parametric function must return a single number.".to_string(),
					)]);
				},
				Ok(Value::Boolean(_)) => {
					return Err(vec![(id, "Non-parametric function must return a number.".to_string())]);
				},
				Err(e) => {
					return Err(vec![(id, e.to_string())]);
				},
			}

			let prev_sampling_arg = sampling_arg;
			sampling_arg += step_size;
			if sampling_arg == prev_sampling_arg {
				break;
			}
		}
	}

	// println!("fn {} num points", pp_buffer.len());
	add_line(pp_buffer);

	Ok(())
}

const P_CURVATURE_FACTOR_SCALE: f64 = 8.0;
const P_SMALL_DISTANCE_SCALE: f64 = 35.0;
const P_SMALL_DISTANCE_POW: f64 = 1.0;
fn calculate_curvature_factor(pangle1: f64, pangle2: f64) -> f64 {
	let mut angle_diff = (pangle2 - pangle1).abs();
	if angle_diff > 2.0 {
		angle_diff = 4.0 - angle_diff;
	}

	// Convert to curvature factor: 1.0 = straight, increases with curvature
	1.0 + (angle_diff * P_CURVATURE_FACTOR_SCALE)
}
#[allow(clippy::too_many_arguments)]
fn draw_parametric_function<T: EvalexprFloat, const TY: SimpleFunctionType>(
	tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>, ctx: &HashMapContext<T>, plot_params: &PlotParams,
	id: u64, func: &ExpressionFunction<T>, range_start: Option<&FlatNode<T>>, range_end: Option<&FlatNode<T>>,
	mut add_line: impl FnMut(Vec<PlotPoint>, bool),
) -> Result<(), Vec<(u64, String)>> {
	let mut stack = thread_local_get(tl_context).stack.borrow_mut();
	let mut pp_buffer = vec![];
	let (start, end) = match eval_point(&mut stack, ctx, range_start, range_end) {
		Ok(Some((start, end))) => (start, end),
		Err(e) => {
			return Err(vec![(id, e)]);
		},
		_ => {
			return Ok(());
		},
	};
	if start > end {
		return Err(vec![(id, "Range start must be less than range end".to_string())]);
	}
	let range = end - start;
	let step = range / plot_params.resolution as f64;
	if start + step == end {
		return Ok(());
	}

	let plot_width = plot_params.last_x - plot_params.first_x;
	let plot_height = plot_params.last_y - plot_params.first_y;

	let small_distance_x = (plot_params.eps * 10_000.0).max(plot_width / plot_params.resolution as f64);
	let small_distance_y = (plot_params.eps * 10_000.0).max(plot_height / plot_params.resolution as f64);
	let max_segment_x = small_distance_x.powf(P_SMALL_DISTANCE_POW) * P_SMALL_DISTANCE_SCALE;
	let max_segment_y = small_distance_y.powf(P_SMALL_DISTANCE_POW) * P_SMALL_DISTANCE_SCALE;
	// println!(
	// 	"small distance x: {small_distance_x} plot eps * 10_000 {} max segment x {max_segment_x}",
	// 	plot_params.eps * 10_000.0
	// );

	#[allow(clippy::integer_division)]
	let max_subdivisions = (plot_params.resolution / 30).clamp(1, 50);

	// Determine function type from first evaluation
	let first_value = match func.call(&mut stack, ctx, &[f64_to_value::<T>(start)]) {
		Ok(v) => v,
		Err(e) => return Err(vec![(id, e.to_string())]),
	};

	let is_float2 = matches!(first_value, Value::Float2(_, _));

	let step_eps = if is_float2 {
		max_segment_y
	} else if TY == SimpleFunctionType::X {
		max_segment_x
	} else {
		max_segment_y
	};
	let mut discontinuity_detector = DiscontinuityDetector::new(step_eps, plot_params.eps);

	let mut eval_point_at = |arg: f64| -> Result<Option<(f64, PlotPoint)>, String> {
		match func.call(&mut stack, ctx, &[f64_to_value::<T>(arg)]) {
			Ok(Value::Float(value)) => {
				if is_float2 {
					return Err("Function must consistently return Float2".to_string());
				}
				if value.is_nan() {
					Ok(None)
				} else {
					let v = value.to_f64();
					match TY {
						SimpleFunctionType::X => Ok(Some((v, PlotPoint::new(arg, v)))),
						SimpleFunctionType::Y => Ok(Some((v, PlotPoint::new(v, arg)))),
					}
				}
			},
			Ok(Value::Empty) => Ok(None),
			Ok(Value::Float2(x, y)) => {
				if !is_float2 {
					return Err("Function must consistently return Float".to_string());
				}
				if y.is_nan() {
					Ok(None)
				} else {
					// Use y value for discontinuity detection
					Ok(Some((y.to_f64(), PlotPoint::new(x.to_f64(), y.to_f64()))))
				}
			},
			Ok(_) => Err("Ranged function must return 1 or 2 float values".to_string()),
			Err(e) => Err(e.to_string()),
		}
	};

	let mut prev_angle: Option<f64> = None;
	// Evaluate first point
	let mut prev_arg = start;
	let mut prev_point = match eval_point_at(start) {
		Ok(r) => r.map(|(_, p)| p),
		Err(e) => return Err(vec![(id, e)]),
	};

	if let Some(p) = prev_point {
		pp_buffer.push(p);
	}

	// let mut num_splits = Vec::with_capacity(plot_params.resolution * 2);
	// let mut curvature_factors = Vec::with_capacity(plot_params.resolution * 2);
	for i in 1..=plot_params.resolution {
		let curr_arg = start + step * i as f64;
		let curr_result = match eval_point_at(curr_arg) {
			Ok(r) => r,
			Err(e) => return Err(vec![(id, e)]),
		};
		let mut cur_angle = None;

		match (prev_point, curr_result) {
			(Some(prev_p), Some((curr_val, curr_p))) => {
				// Check for discontinuity
				if !is_float2
					&& let Some((left, right)) = discontinuity_detector.detect(curr_arg, curr_val, |arg| {
						eval_point_at(arg).ok().and_then(|v| v.map(|(v, _)| v))
					}) {
					if is_float2 {
						pp_buffer.push(PlotPoint::new(left.1, left.0));
						add_line(mem::take(&mut pp_buffer), true);
						pp_buffer.push(PlotPoint::new(right.1, right.0));
					} else {
						match TY {
							SimpleFunctionType::X => {
								pp_buffer.push(PlotPoint::new(left.0, left.1));
								add_line(mem::take(&mut pp_buffer), true);
								pp_buffer.push(PlotPoint::new(right.0, right.1));
							},
							SimpleFunctionType::Y => {
								pp_buffer.push(PlotPoint::new(left.1, left.0));
								add_line(mem::take(&mut pp_buffer), true);
								pp_buffer.push(PlotPoint::new(right.1, right.0));
							},
						}
					}
					prev_arg = curr_arg;

					prev_angle = None;
					prev_point = curr_result.map(|(_, v)| v);
					continue;
				}

				// Check if segment is too long and needs subdivision
				let dx = curr_p.x - prev_p.x;
				let dy = curr_p.y - prev_p.y;
				let a_dx = dx.abs();
				let a_dy = dy.abs();

				let margin_x = plot_width * 0.5 + a_dx * 2.0;
				let margin_y = plot_height * 0.5 + a_dy * 2.0;
				let on_screen = (prev_p.x >= plot_params.first_x - margin_x
					&& prev_p.x <= plot_params.last_x + margin_x
					&& prev_p.y >= plot_params.first_y - margin_y
					&& prev_p.y <= plot_params.last_y + margin_y)
					|| (curr_p.x >= plot_params.first_x - margin_x
						&& curr_p.x <= plot_params.last_x + margin_x
						&& curr_p.y >= plot_params.first_y - margin_y
						&& curr_p.y <= plot_params.last_y + margin_y);

				cur_angle = Some(pseudoangle(dx, dy));
				if on_screen {
					let subdivisions_x =
						if a_dx > max_segment_x { (a_dx / max_segment_x).ceil() as usize } else { 1 };
					let subdivisions_y =
						if a_dy > max_segment_y { (a_dy / max_segment_y).ceil() as usize } else { 1 };
					let mut subdivisions = subdivisions_x.max(subdivisions_y).min(max_subdivisions);

					// Adjust by curvature
					if let Some(prev_angle) = prev_angle {
						let curvature_factor = calculate_curvature_factor(prev_angle, cur_angle.unwrap());

						// 					let segment_length = (dx.powi(2) + dy.powi(2)).sqrt();
						// 					let length_scale = (max_segment_x.max(max_segment_y) /
						// segment_length).clamp(0.01, 1.0);

						// 					let adjusted_curvature = curvature_factor * length_scale.sqrt();
						let adjusted_curvature = curvature_factor;
						subdivisions =
							((subdivisions as f64 * adjusted_curvature) as usize).min(max_subdivisions);

						// curvature_factors.push(adjusted_curvature);
					}

					subdivisions = subdivisions.max(1);

					if subdivisions > 1 {
						// Need to subdivide this segment
						let t_step = (curr_arg - prev_arg) / subdivisions as f64;
						for j in 1..subdivisions {
							let t = prev_arg + t_step * j as f64;
							match eval_point_at(t) {
								Ok(Some((_, p))) => pp_buffer.push(p),
								Ok(None) => {
									// NaN in the middle
									add_line(mem::take(&mut pp_buffer), true);
								},
								Err(e) => return Err(vec![(id, e)]),
							}
						}
					}
					// num_splits.push(subdivisions);
				} else {
					// num_splits.push(1);
				}
				pp_buffer.push(curr_p);
			},
			(Some(_), None) => {
				// nan
				add_line(mem::take(&mut pp_buffer), true);
			},
			(None, Some((_, p))) => {
				// coming back from NaN
				pp_buffer.push(p);
			},
			(None, None) => {
				// Still NaN, nothing to do
			},
		}

		prev_angle = cur_angle;
		prev_arg = curr_arg;
		prev_point = curr_result.map(|(_, p)| p);
	}

	// if num_splits.len() > 0 {
	// num_splits.sort_unstable();
	// let avg_num_splits = num_splits.iter().sum::<usize>() as f64 / num_splits.len() as f64;
	// let total_num_splits = num_splits.iter().sum::<usize>();
	// println!(
	// 	"Average number of splits: {avg_num_splits} median {} max {} total {}",
	// 	num_splits[num_splits.len() / 2],
	// 	num_splits.last().unwrap(),
	// 	total_num_splits
	// );
	// }

	// if curvature_factors.len() > 0 {
	// 	curvature_factors.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
	// 	let avg_curvature_factor =
	// 		curvature_factors.iter().sum::<f64>() as f64 / curvature_factors.len() as f64;
	// 	let total_curvature_factor = curvature_factors.iter().sum::<f64>();
	// 	println!(
	// 		"Average curvature factor: {avg_curvature_factor} min {} median {} max {} total {}",
	// 		curvature_factors[0],
	// 		curvature_factors[curvature_factors.len() / 2],
	// 		curvature_factors.last().unwrap(),
	// 		total_curvature_factor
	// 	);
	// }
	add_line(pp_buffer, false);
	Ok(())
}
#[allow(clippy::too_many_arguments)]
fn draw_implicit<T: EvalexprFloat>(
	tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>, plot_params: &PlotParams, id: u64,
	resolution: usize, equation_type: EquationType, color: Color32,
	call: impl Fn(&mut RefMut<Stack<T>>, f64, f64) -> Result<Value<T>, EvalexprError<T>> + Sync,
	mut add_line: impl FnMut(Vec<PlotPoint>), mut add_mesh: impl FnMut(MeshBuilder),
) -> Result<(), Vec<(u64, String)>> {
	let mins = (plot_params.first_x, plot_params.first_y);
	let maxs = (plot_params.last_x, plot_params.last_y);

	let mut draw_lines = true;
	let mut draw_fill = None;
	match equation_type {
		EquationType::Equality => {},
		EquationType::LessThan => {
			draw_lines = false;
			draw_fill = Some(marching_squares::MarchingSquaresFill::Negative);
		},
		EquationType::LessThanOrEqual => {
			draw_fill = Some(marching_squares::MarchingSquaresFill::Negative);
		},
		EquationType::GreaterThan => {
			draw_lines = false;
			draw_fill = Some(marching_squares::MarchingSquaresFill::Positive);
		},
		EquationType::GreaterThanOrEqual => {
			draw_fill = Some(marching_squares::MarchingSquaresFill::Positive);
		},
		EquationType::None => {
			return Ok(());
		},
	}

	let params = marching_squares::MarchingSquaresParams {
		resolution,
		bounds_min: mins,
		bounds_max: maxs,
		draw_lines,
		draw_fill,
		fill_color: color,
	};
	for result in marching_squares::marching_squares(
		params,
		|cc: &mut RefMut<Stack<T>>, x, y| {
			let value = call(cc, x, y);
			// let value = match TY {
			// 	SimpleFunctionType::X => func.call(&mut *cc, ctx, &[f64_to_value::<T>(x)]),
			// 	SimpleFunctionType::Y => func.call(&mut *cc, ctx, &[f64_to_value::<T>(y)]),
			// };
			match value {
				Ok(v) => match v {
					Value::Float(v) => Ok(v.to_f64()),
					Value::Empty => Ok(f64::NAN),
					Value::Tuple(_) | Value::Float2(_, _) => {
						Err("Implicit function must return a single value.".to_string())
					},
					Value::Boolean(_) => Err("Implicit function must return a number.".to_string()),
				},
				Err(EvalexprError::ExpectedFloat { actual }) => {
					if actual == Value::Empty {
						Ok(f64::NAN)
					} else {
						Err(EvalexprError::ExpectedFloat { actual }.to_string())
					}
				},
				Err(e) => Err(e.to_string()),
			}
		},
		|| {
			let tl_context = thread_local_get(tl_context);
			// tl_context.stack_overflow_guard.set(0);
			tl_context.stack.borrow_mut()
		},
		&thread_local_get(tl_context).marching_squares_cache,
	)
	.map_err(|e| vec![(id, e)])?
	{
		for line in result.lines {
			add_line(line);
		}
		add_mesh(result.mesh);
	}

	Ok(())
}

pub fn gpu_position_from_point(plot_trans: &PlotTransform, invert_axes: [bool; 2], point: &PlotPoint) -> Pos2 {
	let x = remap(
		point.x,
		plot_trans.bounds().min()[0]..=plot_trans.bounds().max()[0],
		if invert_axes[0] { 1.0..=-1.0 } else { -1.0..=1.0 },
	) as f32;

	let y = remap(
		point.y,
		plot_trans.bounds().min()[1]..=plot_trans.bounds().max()[1],
		if invert_axes[1] { 1.0..=-1.0 } else { -1.0..=1.0 },
	) as f32;
	pos2(x, y)
}
