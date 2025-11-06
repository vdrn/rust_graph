use alloc::sync::Arc;
use core::ops::RangeInclusive;
use core::sync::atomic::{AtomicIsize, Ordering};

use eframe::egui::containers::menu::MenuButton;
use eframe::egui::{self, Align, Button, Color32, DragValue, Id, RichText, Slider, Stroke, TextEdit, Widget};
use egui_plot::{Line, PlotPoint, PlotPoints, Points, Polygon};
use evalexpr::{
	ContextWithMutableFunctions, ContextWithMutableVariables, EvalexprError, EvalexprFloat, EvalexprNumericTypes, Function, Node, Value
};
use serde::{Deserialize, Serialize};

use crate::MAX_FUNCTION_NESTING;

pub const COLORS: &[Color32; 20] = &[
	Color32::from_rgb(255, 107, 107), // Bright coral red
	Color32::from_rgb(78, 205, 196),  // Turquoise
	Color32::from_rgb(69, 183, 209),  // Sky blue
	Color32::from_rgb(255, 160, 122), // Light salmon
	Color32::from_rgb(152, 216, 200), // Mint green
	Color32::from_rgb(255, 217, 61),  // Golden yellow
	Color32::from_rgb(107, 207, 127), // Fresh green
	Color32::from_rgb(199, 125, 255), // Bright purple
	Color32::from_rgb(255, 133, 161), // Pink
	Color32::from_rgb(93, 173, 226),  // Bright blue
	Color32::from_rgb(248, 183, 57),  // Orange
	Color32::from_rgb(127, 219, 255), // Aqua
	Color32::from_rgb(57, 255, 20),   // Neon green
	Color32::from_rgb(255, 20, 147),  // Deep pink
	Color32::from_rgb(0, 217, 255),   // Cyan
	Color32::from_rgb(255, 179, 71),  // Peach
	Color32::from_rgb(139, 92, 246),  // Violet
	Color32::from_rgb(52, 211, 153),  // Emerald
	Color32::from_rgb(244, 114, 182), // Hot pink
	Color32::from_rgb(251, 191, 36),  // Amber
];
pub const NUM_COLORS: usize = COLORS.len();

#[derive(Clone, Debug)]
pub struct Entry<T: EvalexprNumericTypes> {
	pub name:    String,
	pub visible: bool,
	pub color:   usize,
	pub ty:      EntryType<T>,
}
#[derive(Clone, Debug)]
pub enum EntryType<T: EvalexprNumericTypes> {
	Function {
		text: String,
		func: Option<Node<T>>,
	},
	Constant {
		value: T::Float,
		step:  f64,
		ty:    ConstantType,
	},
	Points(Vec<PointEntry<T>>),
	Integral {
		func_text:  String,
		func:       Option<Node<T>>,
		lower_text: String,
		lower:      Option<Node<T>>,
		upper_text: String,
		upper:      Option<Node<T>>,
		calculated: Option<T::Float>,
		resolution: usize,
	},
}

impl<T: EvalexprNumericTypes> Entry<T> {
	pub fn symbol(&self) -> &'static str {
		match self.ty {
			EntryType::Function { .. } => "λ",
			EntryType::Constant { .. } => {
				if self.visible {
					"⏸"
				} else {
					"⏵"
				}
			},
			EntryType::Points(_) => "●",
			EntryType::Integral { .. } => "∫",
		}
	}
	pub fn color(&self) -> Color32 { COLORS[self.color % NUM_COLORS] }
	pub fn new_function(color: usize, text: String) -> Self {
		Self {
			color:   color % NUM_COLORS,
			visible: true,
			name:    String::new(),
			ty:      EntryType::Function { text, func: None },
		}
	}
	pub fn new_constant(color: usize) -> Self {
		Self {
			color:   color % NUM_COLORS,
			visible: false,
			name:    String::new(),
			ty:      EntryType::Constant {
				value: T::Float::ZERO,
				step:  0.01,
				ty:    ConstantType::LoopForwardAndBackward { start: 0.0, end: 2.0, forward: true },
			},
		}
	}
	pub fn new_points(color: usize) -> Self {
		Self {
			color:   color % NUM_COLORS,
			visible: true,
			name:    String::new(),
			ty:      EntryType::Points(vec![PointEntry::default()]),
		}
	}
	pub fn new_integral(color: usize) -> Self {
		Self {
			color:   color % NUM_COLORS,
			visible: true,
			name:    String::new(),
			ty:      EntryType::Integral {
				func_text:  String::new(),
				func:       None,
				lower_text: String::new(),
				lower:      None,
				upper_text: String::new(),
				upper:      None,
				calculated: None,
				resolution: 500,
			},
		}
	}
}

#[derive(Clone, Debug)]
pub struct PointEntry<T: EvalexprNumericTypes> {
	pub text_x: String,
	pub x:      Option<Node<T>>,
	pub text_y: String,
	pub y:      Option<Node<T>>,
}

impl<T: EvalexprNumericTypes> Default for PointEntry<T> {
	fn default() -> Self {
		Self {
			text_x: Default::default(),
			x:      Default::default(),
			text_y: Default::default(),
			y:      Default::default(),
		}
	}
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConstantType {
	LoopForwardAndBackward { start: f64, end: f64, forward: bool },
	LoopForward { start: f64, end: f64 },
	PlayOnce { start: f64, end: f64 },
	PlayIndefinitely { start: f64 },
}
impl ConstantType {
	pub fn range(&self) -> RangeInclusive<f64> {
		match self {
			ConstantType::LoopForwardAndBackward { start, end, .. } => *start..=*end,
			ConstantType::LoopForward { start, end } => *start..=*end,
			ConstantType::PlayOnce { start, end } => *start..=*end,
			ConstantType::PlayIndefinitely { start } => *start..=f64::INFINITY,
		}
	}
	pub fn symbol(&self) -> &'static str {
		match self {
			ConstantType::LoopForwardAndBackward { .. } => "🔁",
			ConstantType::LoopForward { .. } => "🔂",
			ConstantType::PlayOnce { .. } => "⏯",
			ConstantType::PlayIndefinitely { .. } => "🔀",
		}
	}
	pub fn name(&self) -> &'static str {
		match self {
			ConstantType::LoopForwardAndBackward { .. } => "🔁 Loop forward and Backward",
			ConstantType::LoopForward { .. } => "🔂 Loop forward",
			ConstantType::PlayOnce { .. } => "⏯ Play once",
			ConstantType::PlayIndefinitely { .. } => "🔀 Play indefinitely",
		}
	}
}

pub struct EditEntryResult {
	pub needs_recompilation: bool,
	pub animating:           bool,
	pub remove:              bool,
	pub error:               Option<String>,
}
pub fn edit_entry_ui<T: EvalexprNumericTypes>(
	ui: &mut egui::Ui, entry: &mut Entry<T>, clear_cache: bool,
) -> EditEntryResult {
	let mut result = EditEntryResult {
		needs_recompilation: false,
		animating:           false,
		remove:              false,
		error:               None,
	};

	let (text_col, fill_col) = if entry.visible {
		(Color32::BLACK, entry.color())
	} else {
		(Color32::LIGHT_GRAY, egui::Color32::TRANSPARENT)
	};

	ui.with_layout(egui::Layout::right_to_left(Align::LEFT), |ui| {
		if ui.button("X").clicked() {
			result.remove = true;
			result.needs_recompilation = true;
		}
		let mut color_picker = MenuButton::new(RichText::new("🎨").color(Color32::BLACK));
		color_picker.button = color_picker.button.fill(entry.color());
		color_picker.ui(ui, |ui| {
			for i in 0..COLORS.len() {
				if ui.button(RichText::new("     ").background_color(COLORS[i])).clicked() {
					entry.color = i;
				}
			}
		});
		ui.with_layout(egui::Layout::left_to_right(Align::LEFT), |ui| {
			let prev_visible = entry.visible;
			if ui
				.add(
					Button::new(RichText::new(entry.symbol()).strong().monospace().color(text_col))
						.fill(fill_col)
						.corner_radius(10),
				)
				.clicked()
			{
				entry.visible = !entry.visible;
			}

			if ui.add(TextEdit::singleline(&mut entry.name).desired_width(30.0).hint_text("name")).changed() {
				result.needs_recompilation = true;
			}
			let color = entry.color();
			match &mut entry.ty {
				EntryType::Function { text, func } => {
					match edit_expr(ui, text, func, "sin(x)", None, clear_cache) {
						Ok(changed) => {
							result.needs_recompilation |= changed;
						},
						Err(e) => {
							result.error = Some(format!("Parsing error: {e}"));
						},
					}
				},
				EntryType::Integral {
					func,
					lower,
					upper,
					func_text,
					lower_text,
					upper_text,
					calculated,
					resolution,
				} => {
					ui.vertical(|ui| {
						ui.horizontal(|ui| {
							ui.label("Lower:");
							match edit_expr(ui, lower_text, lower, "lower", Some(30.0), clear_cache) {
								Ok(_changed) => {
									// needs_recompilation |= changed;
								},
								Err(e) => {
									result.error = Some(format!("Parsing error: {e}"));
								},
							}
							ui.label("Upper:");
							match edit_expr(ui, upper_text, upper, "upper", Some(30.0), clear_cache) {
								Ok(_changed) => {
									// needs_recompilation |= changed;
								},
								Err(e) => {
									result.error = Some(format!("Parsing error: {e}"));
								},
							}
						});
						ui.horizontal(|ui| {
							ui.label("Func:");
							match edit_expr(ui, func_text, func, "func", None, clear_cache) {
								Ok(changed) => {
									result.needs_recompilation |= changed;
								},
								Err(e) => {
									result.error = Some(format!("Parsing error: {e}"));
								},
							};
							ui.label("dx");
						});
						if let Some(calculated) = calculated {
							ui.label(RichText::new(format!("Value: {}", calculated)).color(color));
						}

						ui.add(Slider::new(resolution, 10..=1000).text("Resolution"));
					});
				},
				EntryType::Points(points) => {
					let mut remove_point = None;
					ui.vertical(|ui| {
						for (pi, point) in points.iter_mut().enumerate() {
							ui.horizontal(|ui| {
								match edit_expr(
									ui,
									&mut point.text_x,
									&mut point.x,
									"point_x",
									Some(80.0),
									clear_cache,
								) {
									Ok(changed) => {
										result.needs_recompilation |= changed;
									},
									Err(e) => {
										result.error = Some(format!("Parsing error: {e}"));
									},
								}
								match edit_expr(
									ui,
									&mut point.text_y,
									&mut point.y,
									"point_y",
									Some(80.0),
									clear_cache,
								) {
									Ok(changed) => {
										result.needs_recompilation |= changed;
									},
									Err(e) => {
										result.error = Some(format!("Parsing error: {e}"));
									},
								}

								if ui.button("X").clicked() {
									remove_point = Some(pi);
								}
							});
						}
						if let Some(pi) = remove_point {
							points.remove(pi);
						}
						ui.horizontal(|ui| {
							if ui.button("Add Point").clicked() {
								points.push(PointEntry::default());
							}
						});
					});
				},

				EntryType::Constant { value, step, ty } => {
					let mut v = value.to_f64();
					let step_f = T::Float::f64_to_float(*step);
					let range = ty.range();
					let start = *range.start();
					let end = *range.end();
					v = v.clamp(start, end);

					ui.vertical(|ui| {
						ui.horizontal(|ui| {
							ui.menu_button(ty.symbol(), |ui| {
								let new_end = if end.is_infinite() { start + 10.0 } else { end };
								let lfab = ConstantType::LoopForwardAndBackward {
									start,
									end: new_end,
									forward: true,
								};
								if ui.button(lfab.name()).clicked() {
									*ty = lfab;
									result.animating = true;
								}
								let lf = ConstantType::LoopForward { start, end: new_end };
								if ui.button(lf.name()).clicked() {
									*ty = lf;
									result.animating = true;
								}
								let po = ConstantType::PlayOnce { start, end: new_end };
								if ui.button(po.name()).clicked() {
									*ty = po;
									result.animating = true;
								}
								let pi = ConstantType::PlayIndefinitely { start };
								if ui.button(pi.name()).clicked() {
									*ty = pi;
									result.animating = true;
								}
							});
							match ty {
								ConstantType::LoopForwardAndBackward { start, end, .. }
								| ConstantType::LoopForward { start, end }
								| ConstantType::PlayOnce { start, end } => {
									DragValue::new(start).prefix("Start:").speed(*step).ui(ui);
									DragValue::new(end).prefix("End:").speed(*step).ui(ui);
									*start = start.min(*end);
									*end = end.max(*start);
								},
								ConstantType::PlayIndefinitely { start } => {
									DragValue::new(start).prefix("Start:").ui(ui);
								},
							}

							DragValue::new(step).prefix("Step:").speed(0.00001).ui(ui);

							if !prev_visible && entry.visible {
								if value.to_f64() >= end {
									*value = T::Float::f64_to_float(start);
									result.animating = true;
								}
							}

							if entry.visible {
								ui.ctx().request_repaint();
								result.animating = true;

								match ty {
									ConstantType::LoopForwardAndBackward { forward, .. } => {
										if value.to_f64() > end {
											*forward = false;
										}
										if value.to_f64() < start {
											*forward = true;
										}
										if *forward {
											*value = *value + step_f;
										} else {
											*value = *value - step_f;
										}
									},
									ConstantType::LoopForward { .. } => {
										*value = *value + step_f;
										if value.to_f64() >= end {
											*value = T::Float::f64_to_float(start);
										}
									},
									ConstantType::PlayOnce { .. } | ConstantType::PlayIndefinitely { .. } => {
										*value = *value + step_f;
										if value.to_f64() >= end {
											entry.visible = false;
										}
									},
								}
							}
						});
						if ui
							.add(
								Slider::new(&mut v, range)
									.step_by(*step)
									.clamping(egui::SliderClamping::Never),
							)
							.changed() || clear_cache
						{
							*value = T::Float::f64_to_float(v);
							result.animating = true;
						}
					});
				},
			}
		});
	});

	result
}
fn edit_expr<T: EvalexprNumericTypes>(
	ui: &mut egui::Ui, text: &mut String, expr: &mut Option<Node<T>>, hint_text: &str,
	desired_width: Option<f32>, force_update: bool,
) -> Result<bool, String> {
	let mut text_edit = TextEdit::singleline(text).hint_text(hint_text);
	if let Some(width) = desired_width {
		text_edit = text_edit.desired_width(width);
	}
	if ui.add(text_edit).changed() || force_update {
		if text.is_empty() {
			*expr = None;
		} else {
			*expr = match evalexpr::build_operator_tree::<T>(text) {
				Ok(func) => Some(func),
				Err(e) => {
					return Err(e.to_string());
				},
			};
		}

		return Ok(true);
	}
	Ok(false)
}

pub fn recompile_entry<T: EvalexprNumericTypes>(
	entry: &mut Entry<T>, ctx: &Arc<std::sync::RwLock<evalexpr::HashMapContext<T>>>,
	stack_overflow_guard: &Arc<AtomicIsize>,
) {
	puffin::profile_scope!("entry_recompile");
	if entry.name != "x" {
		match &mut entry.ty {
			EntryType::Points(_) => {},
			EntryType::Integral { .. } => {},
			EntryType::Constant { value, .. } => {
				if !entry.name.is_empty() {
					ctx.write()
						.unwrap()
						.set_value(entry.name.as_str(), evalexpr::Value::<T>::Float(*value))
						.unwrap();
				}
			},
			EntryType::Function { text, func } => {
				if let Some(func) = func.clone() {
					// struct LocalCache(Mutex<AHashMap<CacheKey, f64>>);
					// impl Clone for LocalCache {
					// 	fn clone(&self) -> Self {
					// 		Self(Mutex::new(self.0.lock().unwrap().clone()))
					// 	}
					// }
					// impl LocalCache {
					// 	fn lock(&self) -> MutexGuard<'_, AHashMap<CacheKey, f64>> {
					// 		self.0.lock().unwrap()
					// 	}
					// }
					// let local_cache: LocalCache = LocalCache(Mutex::new(
					// 	AHashMap::with_capacity(ui_state.conf.resolution),
					// ));
					// let animating = ui_state.animating.clone();
					let main_context = ctx.clone();
					let stack_overflow_guard = stack_overflow_guard.clone();
					let fun = Function::new(move |v| {
						// puffin::profile_scope!("eval_function");
						if stack_overflow_guard.fetch_add(1, Ordering::Relaxed) > MAX_FUNCTION_NESTING {
							return Err(EvalexprError::CustomMessage(format!(
								"Max function nesting reached ({MAX_FUNCTION_NESTING})"
							)));
						}

						let v = match v {
							Value::Float(x) => *x,
							Value::Boolean(x) => T::Float::f64_to_float(*x as i64 as f64),
							Value::String(_) => T::Float::ZERO,
							// Value::Int(x) => T::int_as_float(x),
							Value::Tuple(values) => values[0]
								.as_float()
								.or_else(|_| {
									values[0].as_boolean().map(|x| T::Float::f64_to_float(x as i64 as f64))
								})
								.unwrap_or(T::Float::ZERO),
							Value::Empty => T::Float::ZERO,
						};
						// let animating = animating.load(Ordering::Relaxed);
						// if !animating {
						// 	if let Some(cached) =
						// 		local_cache.lock().get(&CacheKey(T::float_to_f64(v)))
						// 	{
						// 		stack_overflow_guard
						// 			.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
						// 		return Ok(Value::Float(T::f64_to_float(*cached)));
						// 	}
						// }
						let vv = Value::<T>::Float(v);

						let context = main_context.read().unwrap();

						let res = { func.eval_float_with_context_and_x(&*context, &vv) };
						stack_overflow_guard.fetch_sub(1, Ordering::Relaxed);
						res.map(|res| {
							// if !animating {
							// 	local_cache.lock().insert(
							// 		CacheKey(T::float_to_f64(v)),
							// 		T::float_to_f64(res),
							// 	);
							// }
							Value::Float(res)
						})
					});

					let name = if entry.name.is_empty() { text.clone() } else { entry.name.clone() };

					ctx.write().unwrap().set_function(name, fun).unwrap();
				}
			},
		}
	}
}

pub struct PlotParams {
	pub eps:       f64,
	pub first_x:   f64,
	pub last_x:    f64,
	pub step_size: f64,
}
pub fn create_entry_plot_elements<T: EvalexprNumericTypes>(
	entry: &mut Entry<T>, id: Id, selected: bool, ctx: &evalexpr::HashMapContext<T>, plot_params: &PlotParams,
	polygons: &mut Vec<Polygon<'static>>, lines: &mut Vec<(bool, Id, Line<'static>)>,
	points: &mut Vec<Points<'static>>,
) -> Result<(), String> {
	let visible = entry.visible;
	if !visible && !matches!(entry.ty, EntryType::Integral { .. }) {
		return Ok(());
	}
	puffin::profile_scope!("entry_draw", entry.name.clone());
	let color = entry.color();

	match &mut entry.ty {
		EntryType::Constant { .. } => {},
		EntryType::Integral { func, func_text, lower, upper, calculated, resolution, .. } => {
			let (Some(lower), Some(upper), Some(func)) = (lower, upper, func) else {
				return Ok(());
			};
			let lower = match lower.eval_float_with_context(ctx) {
				Ok(lower) => lower.to_f64(),
				Err(e) => {
					return Err(format!("Error evaluating lower bound: {e}"));
				},
			};
			let upper = match upper.eval_float_with_context(ctx) {
				Ok(upper) => upper.to_f64(),
				Err(e) => {
					return Err(format!("Error evaluating upper bound: {e}"));
				},
			};
			let range = upper - lower;
			if lower > upper {
				*calculated = None;

				return Err("Lower bound must be less than upper bound".to_string());
			}
			let resolution = *resolution;
			let step = range / resolution as f64;
			let step_f = T::Float::f64_to_float(step);
			if lower + step == lower {
				*calculated = Some(T::Float::ZERO);
				return Ok(());
			}
			*calculated = None;

			let mut int_lines = vec![];
			let mut fun_lines = vec![];
			let stroke_color = color;
			let rgba_color = stroke_color.to_srgba_unmultiplied();
			let fill_color = Color32::from_rgba_unmultiplied(rgba_color[0], rgba_color[1], rgba_color[1], 128);

			let mut result: T::Float = T::Float::ZERO;
			let mut prev_y: Option<T::Float> = None;
			for i in 0..(resolution + 1) {
				let x = lower + step * i as f64;

				let xx = evalexpr::Value::<T>::Float(T::Float::f64_to_float(x));

				let y_f64 = match func.eval_float_with_context_and_x(ctx, &xx) {
					Ok(y) => y.to_f64(),
					Err(e) => {
						return Err(e.to_string());
					},
				};

				let y = T::Float::f64_to_float(y_f64);
				if let Some(prev_y) = prev_y {
					let eps = 0.0;
					let prev_y_f64 = prev_y.to_f64();

					if prev_y_f64.signum() != y_f64.signum() {
						//2 triangles
						let diff = (prev_y_f64 - y_f64).abs();
						let t = prev_y_f64.abs() / diff;
						let x_midpoint = (x - step) + step * t;
						if visible {
							let triangle1 = Polygon::new(
								entry.name.clone(),
								PlotPoints::Owned(vec![
									PlotPoint::new(x - step, 0.0),
									PlotPoint::new(x - step, prev_y_f64),
									PlotPoint::new(x_midpoint, 0.0),
								]),
							)
							.fill_color(fill_color)
							.stroke(Stroke::new(eps, fill_color));
							polygons.push(triangle1);
							let triangle2 = Polygon::new(
								entry.name.clone(),
								PlotPoints::Owned(vec![
									PlotPoint::new(x_midpoint, 0.0),
									PlotPoint::new(x, y_f64),
									PlotPoint::new(x, 0.0),
								]),
							)
							.fill_color(fill_color)
							.stroke(Stroke::new(eps, fill_color));
							polygons.push(triangle2);
						}

						let t = T::Float::f64_to_float(t);

						let step1 = step_f * t;
						let step2 = step_f - step1;

						let b1 = prev_y * step1;
						let b2 = y * step2;
						result = result + b1 * T::Float::HALF;
						result = result + b2 * T::Float::HALF;
					} else {
						if visible {
							let poly = Polygon::new(
								entry.name.clone(),
								if prev_y_f64 > 0.0 {
									PlotPoints::Owned(vec![
										PlotPoint::new(x - step, 0.0),
										PlotPoint::new(x - step, prev_y_f64),
										PlotPoint::new(x, y_f64),
										PlotPoint::new(x, 0.0),
									])
								} else {
									PlotPoints::Owned(vec![
										PlotPoint::new(x - step, 0.0),
										PlotPoint::new(x, 0.0),
										PlotPoint::new(x, y_f64),
										PlotPoint::new(x - step, prev_y_f64),
									])
								},
							)
							.fill_color(fill_color)
							.stroke(Stroke::new(eps, fill_color));
							polygons.push(poly);
						}
						let dy = y - prev_y;
						let step = T::Float::f64_to_float(step);
						let d = dy * step;
						result = result + prev_y * step;
						result = result + d * T::Float::HALF;
					}
				}
				if result.is_nan() {
					return Err("Integral is undefined".to_string());
				}

				if visible {
					int_lines.push(PlotPoint::new(x, result.to_f64()));
					fun_lines.push(PlotPoint::new(x, y_f64));
				}
				prev_y = Some(y);
				// x += step;
			}
			let int_name = if entry.name.is_empty() { func_text.as_str() } else { entry.name.as_str() };
			lines.push((
				false,
				Id::NULL,
				Line::new(format!("∫[{},x]({})dx", lower, int_name), PlotPoints::Owned(int_lines))
					.color(stroke_color),
			));
			lines.push((false, Id::NULL, Line::new("", PlotPoints::Owned(fun_lines)).color(stroke_color)));
			*calculated = Some(result);
		},
		EntryType::Points(ps) => {
			// main_context
			// 	.write()
			// 	.unwrap()
			// 	.set_value("x", evalexpr::Value::<T>::Float(T::ZERO))
			// 	.unwrap();
			let mut line_buffer = vec![];
			for p in ps {
				match eval_point(ctx, p) {
					Ok(Some((x, y))) => line_buffer.push([x, y]),
					Err(e) => {
						return Err(e);
					},
					_ => {},
				}
			}

			if line_buffer.len() == 1 {
				let radius = if selected { 5.0 } else { 3.5 };
				points
					.push(Points::new(entry.name.clone(), line_buffer[0]).id(id).color(color).radius(radius));
			} else if line_buffer.len() > 1 {
				let line = Line::new(entry.name.clone(), line_buffer).color(color).id(id).width(if selected {
					3.5
				} else {
					1.0
				});
				lines.push((false, id, line));
				// plot_ui.line(line);
			}
		},
		EntryType::Function { text, func } => {
			if let Some(func) = func {
				let name = if entry.name.is_empty() {
					format!("y = {}", text)
				} else {
					format!("{}(x) = {}", entry.name, text)
				};
				// let mut cache = (!animating).then(|| {
				// 	state
				// 		.points_cache
				// 		.entry(text.clone())
				// 		.or_insert_with(|| AHashMap::with_capacity(ui_state.conf.resolution))
				// });

				let mut local_optima_buffer = vec![];
				let mut pp_buffer = vec![];

				let mut x = plot_params.first_x;
				let mut prev: [Option<(f64, f64)>; 2] = [None; 2];
				while x < plot_params.last_x {
					let cur_x = x;
					puffin::profile_scope!("graph_step");

					let cur_y =
                // if let Some(cache) = &mut cache {
								// let value = cache.entry(CacheKey(x));

								// match value {
									// hash_map::Entry::Occupied(entry) => entry.get().clone(),
									// hash_map::Entry::Vacant(entry) => {
										// let x = evalexpr::Value::<T>::Float(T::f64_to_float(x));
										// match func
											// .eval_number_with_context_and_x(&*main_context.read().unwrap(), &x)
										// {
											// Ok(y) => {
												// entry.insert(T::float_to_f64(y));
												// T::float_to_f64(y)
											// },
											// Err(e) => {
												// ui_state.eval_errors.insert(ei, e.to_string());

												// continue 'next_entry;
											// },
										// }
									// },
								// }
							// } else
                {
								let x = evalexpr::Value::<T>::Float(T::Float::f64_to_float(x));
								match func.eval_float_with_context_and_x(ctx, &x) {
									Ok(y) => y.to_f64(),
									Err(e) => {
										return Err(e.to_string());

									},
								}
							};
					pp_buffer.push(PlotPoint::new(x, cur_y));
					let cur = (cur_x, cur_y);
					fn less_then(a: f64, b: f64, e: f64) -> bool { b - a > e }
					fn greater_then(a: f64, b: f64, e: f64) -> bool { a - b > e }
					if selected {
						if let (Some(prev_0), Some(prev_1)) = (prev[0], prev[1]) {
							if less_then(prev_0.1, prev_1.1, plot_params.eps)
								&& greater_then(prev_1.1, cur.1, plot_params.eps)
							{
								local_optima_buffer.push(
									Points::new("", [prev_1.0, prev_1.1]).color(Color32::GRAY).radius(3.5),
								);
							}
							if greater_then(prev_0.1, prev_1.1, plot_params.eps)
								&& less_then(prev_1.1, cur.1, plot_params.eps)
							{
								local_optima_buffer.push(
									Points::new("", [prev_1.0, prev_1.1]).color(Color32::GRAY).radius(3.5),
								);
							}
						}
						prev[0] = prev[1];
						prev[1] = Some(cur);
					}

					let prev_x = x;
					x += plot_params.step_size;
					if x == prev_x {
						break;
					}
				}

				let line = Line::new(name, PlotPoints::Owned(pp_buffer))
					.id(id)
					.width(if selected { 3.5 } else { 1.0 })
					.color(entry.color());
				lines.push((true, id, line));

				for point in local_optima_buffer {
					points.push(point);
				}
			}
		},
	}
	Ok(())
}

fn eval_point<T: EvalexprNumericTypes>(
	ctx: &evalexpr::HashMapContext<T>, p: &PointEntry<T>,
) -> Result<Option<(f64, f64)>, String> {
	let (Some(x), Some(y)) = (&p.x, &p.y) else {
		return Ok(None);
	};
	let x = x.eval_float_with_context_and_x(ctx, &Value::Float(T::Float::ZERO)).map_err(|e| e.to_string())?;
	let y = y.eval_float_with_context_and_x(ctx, &Value::Float(T::Float::ZERO)).map_err(|e| e.to_string())?;
	Ok(Some((x.to_f64(), y.to_f64())))
}
