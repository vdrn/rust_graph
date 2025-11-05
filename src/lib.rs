use core::ops::RangeInclusive;
use core::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::collections::hash_map;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::{env, fs};

use ahash::AHashMap;
use eframe::egui::containers::menu::MenuButton;
use eframe::egui::{
	self, Align, Button, CollapsingHeader, DragValue, Grid, Id, RichText, ScrollArea, SidePanel, Slider, Stroke, TextEdit, TextStyle, Visuals, Widget
};
use eframe::epaint::Color32;
use eframe::{App, Storage};
use egui_plot::{
	HLine, Legend, Line, Plot, PlotBounds, PlotGeometry, PlotItem, PlotPoint, PlotPoints, Points, Polygon, VLine
};
use evalexpr::{
	Context, ContextWithMutableFunctions, ContextWithMutableVariables, DefaultNumericTypes, EvalexprError, EvalexprFloat, EvalexprNumericTypes, Function, Node, Value
};
use serde::{Deserialize, Serialize};

mod f32_numeric_type;

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::{self, prelude::*};
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_main() -> () {
	use eframe::wasm_bindgen::JsCast as _;

	let web_options = eframe::WebOptions::default();

	console_error_panic_hook::set_once();

	wasm_bindgen_futures::spawn_local(async {
		let document = web_sys::window().expect("No window").document().expect("No document");

		let canvas = document
			.get_element_by_id("the_canvas_id")
			.expect("Failed to find the_canvas_id")
			.dyn_into::<web_sys::HtmlCanvasElement>()
			.expect("the_canvas_id was not a HtmlCanvasElement");

		let start_result = eframe::WebRunner::new()
			.start(canvas, web_options, Box::new(|cc| Ok(Box::new(Application::new(cc.storage)))))
			.await;

		// Remove the loading text and spinner:
		if let Some(loading_text) = document.get_element_by_id("loading_text") {
			match start_result {
				Ok(_) => {
					loading_text.remove();
				},
				Err(e) => {
					loading_text.set_inner_html(
						"<p> The app has crashed. See the developer console for details. </p>",
					);
					panic!("Failed to start eframe: {e:?}");
				},
			}
		}
	});
}

const DATA_KEY: &str = "data";
const CONF_KEY: &str = "conf";
const MAX_FUNCTION_NESTING: isize = 50;
pub trait NumericTypesExt: EvalexprNumericTypes + Send + Sync
where
	<Self as EvalexprNumericTypes>::Float: Send + Sync, {
	const ZERO: Self::Float;
	const HALF: Self::Float;
	const EPSILON: f64;
	fn float_to_f64(v: Self::Float) -> f64;
	fn f64_to_float(v: f64) -> Self::Float;
	fn float_add(v1: Self::Float, v2: Self::Float) -> Self::Float;
	fn float_mul(v1: Self::Float, v2: Self::Float) -> Self::Float;
	fn float_sub(v1: Self::Float, v2: Self::Float) -> Self::Float;
}
impl NumericTypesExt for DefaultNumericTypes {
	const ZERO: Self::Float = 0.0;
	const HALF: Self::Float = 0.5;
	const EPSILON: f64 = f64::EPSILON;

	fn float_to_f64(v: Self::Float) -> f64 { v }
	fn f64_to_float(v: f64) -> Self::Float { v }
	fn float_add(v1: Self::Float, v2: Self::Float) -> Self::Float { v1 + v2 }
	fn float_mul(v1: Self::Float, v2: Self::Float) -> Self::Float { v1 * v2 }
	fn float_sub(v1: Self::Float, v2: Self::Float) -> Self::Float { v1 - v2 }
}
impl NumericTypesExt for F32NumericType {
	const ZERO: Self::Float = 0.0;
	const HALF: Self::Float = 0.5;
	const EPSILON: f64 = f32::EPSILON as f64;

	fn float_to_f64(v: Self::Float) -> f64 { v as f64 }
	fn f64_to_float(v: f64) -> Self::Float { v as Self::Float }
	fn float_add(v1: Self::Float, v2: Self::Float) -> Self::Float { v1 + v2 }
	fn float_mul(v1: Self::Float, v2: Self::Float) -> Self::Float { v1 * v2 }
	fn float_sub(v1: Self::Float, v2: Self::Float) -> Self::Float { v1 - v2 }
}

use crate::f32_numeric_type::F32NumericType;

const COLORS: &'static [Color32; 20] = &[
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
const NUM_COLORS: usize = COLORS.len();

#[derive(PartialEq, Clone)]
struct CacheKey(f64);
impl core::hash::Hash for CacheKey {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.0.to_bits().hash(state); }
}
impl Eq for CacheKey {}

type PointsCache = ahash::AHashMap<String, ahash::AHashMap<CacheKey, f64>>;

struct State<T: NumericTypesExt>
where
	T::Float: Send + Sync,
	T::Int: Send + Sync, {
	entries:      Vec<Entry<T>>,
	ctx:          &'static std::sync::RwLock<evalexpr::HashMapContext<T>>,
	// context_stash: &'static Mutex<Vec<evalexpr::HashMapContext<T>>>,
	name:         String,
	points_cache: PointsCache,
	clear_cache:  bool,
}
fn init_consts<T: NumericTypesExt>(ctx: &mut evalexpr::HashMapContext<T>)
where
	T::Float: Send + Sync,
	T::Int: Send + Sync, {
	macro_rules! add_const {
		($ident:ident, $uppercase:expr, $lowercase:expr) => {
			ctx.set_value($lowercase, Value::Float(T::f64_to_float(std::f64::consts::$ident))).unwrap();
			ctx.set_value($uppercase, Value::Float(T::f64_to_float(std::f64::consts::$ident))).unwrap();
		};
	}

	add_const!(E, "E", "e");
	add_const!(PI, "PI", "pi");
	add_const!(TAU, "TAU", "tau");
}
fn init_functions<T: NumericTypesExt>(ctx: &mut evalexpr::HashMapContext<T>)
where
	T::Float: Send + Sync,
	T::Int: Send + Sync, {
	macro_rules! add_function {
		($ident:ident) => {
			ctx.set_function(
				stringify!($ident).to_string(),
				evalexpr::Function::new(move |v| {
					let v: T::Float = v.as_number()?;
					Ok(Value::Float(v.$ident()))
				}),
			)
			.unwrap();
		};
	}
	add_function!(ln);
	add_function!(log2);
	add_function!(log10);

	add_function!(exp);
	add_function!(exp2);

	add_function!(cos);
	add_function!(cosh);
	add_function!(acos);
	add_function!(acosh);

	add_function!(sin);
	add_function!(sinh);
	add_function!(asin);
	add_function!(asinh);

	add_function!(tan);
	add_function!(tanh);
	add_function!(atan);
	add_function!(atanh);

	add_function!(sqrt);
	add_function!(cbrt);
	add_function!(asin);

	add_function!(abs);

	macro_rules! add_function_2 {
		($ident:ident) => {
			ctx.set_function(
				stringify!($ident).to_string(),
				evalexpr::Function::new(move |v| {
					let mut v = v.as_fixed_len_tuple(2)?;
					let v2: T::Float = v.pop().unwrap().as_number()?;
					let v1: T::Float = v.pop().unwrap().as_number()?;

					Ok(Value::Float(v1.$ident(&v2)))
				}),
			)
			.unwrap();
		};
	}

	add_function_2!(pow);
	add_function_2!(log);
	add_function_2!(atan2);
	add_function_2!(hypot);
}

#[derive(Serialize, Deserialize)]
struct AppConfig {
	dark_mode:  bool,
	use_f32:    bool,
	resolution: usize,
}
impl Default for AppConfig {
	fn default() -> Self { Self { dark_mode: true, use_f32: false, resolution: 500 } }
}

struct UiState {
	conf:        AppConfig,
	next_color:  usize,
	plot_bounds: PlotBounds,
	// data_aspect: f32,
	reset_graph: bool,

	cur_dir:              String,
	serialization_error:  Option<String>,
	web_storage:          AHashMap<String, String>,
	stack_overflow_guard: &'static AtomicIsize,
	eval_errors:          AHashMap<usize, String>,
	selected_plot_item:   Option<Id>,

	f32_epsilon:          f64,
	f64_epsilon:          f64,
	permalink_string:     String,
	scheduled_url_update: bool,
	last_url_update:      f64,
	animating:            Arc<AtomicBool>,
}
// #[derive(Clone, Debug)]
pub struct Application {
	d:      State<DefaultNumericTypes>,
	s:      State<F32NumericType>,
	ui:     UiState,
	#[cfg(not(target_arch = "wasm32"))]
	puffin: puffin_http::Server,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_puffin_server() -> puffin_http::Server {
	let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
	let puffin_server = puffin_http::Server::new(&server_addr).unwrap();
	println!("Run this to view profiling data:  puffin_viewer {server_addr}");

	puffin::set_scopes_on(true);

	puffin_server
}
impl Application {
	pub fn new(storage: Option<&dyn Storage>) -> Self {
		let mut entries_s = Vec::new();
		let mut entries_d = Vec::new();

		// TODO
		let mut web_storage = AHashMap::new();
		let mut serialization_error = None;

		let conf = storage
			.and_then(|s| s.get_string(CONF_KEY).and_then(|d| serde_json::from_str(&d).ok()))
			.unwrap_or(AppConfig::default());

		cfg_if::cfg_if! {
			  if #[cfg(target_arch = "wasm32")] {
			let cur_dir = String::new();
			// 	let error: Option<String> = web::get_data_from_url(&mut data);
			if let Some(storage) = storage{
			  if let Some(data) = storage.get_string(DATA_KEY) {
				web_storage = serde_json::from_str(&data).unwrap();

			  }

			}

		match deserialize_from_url::<F32NumericType>() {
		  Ok(data)=>{
			entries_s = data;
		  },
		  Err(e)=>{
			serialization_error = Some(e);
		  }
		}
		match deserialize_from_url::<DefaultNumericTypes>() {
		  Ok(data)=>{
			entries_d = data;
		  },
		  Err(e)=>{
			serialization_error = Some(e);
		  }
		}


			  } else {
			let cur_dir = env::home_dir()
				.and_then(|d| d.join("rust_graphs").to_str().map(|s| s.to_string()))
				.unwrap_or_default();
			Self::load_file_entries(&cur_dir, &mut web_storage);
			  }
			}

		let mut next_color = 0;
		if entries_s.is_empty() {
			next_color = 1;
			entries_s.push(Entry::new_function(0, "sin(x)".to_string()));
		}
		let ctx_s = Box::leak(Box::new(RwLock::new(evalexpr::HashMapContext::new())));

		if entries_d.is_empty() {
			next_color = 1;
			entries_d.push(Entry::new_function(0, "sin(x)".to_string()));
		}
		let ctx_d = Box::leak(Box::new(RwLock::new(evalexpr::HashMapContext::new())));

		Self {
			#[cfg(not(target_arch = "wasm32"))]
			puffin:                                     run_puffin_server(),
			s:                                          State {
				entries:      entries_s,
				ctx:          ctx_s,
				name:         String::new(),
				points_cache: PointsCache::default(),
				clear_cache:  true,
			},
			d:                                          State {
				entries:      entries_d,
				ctx:          ctx_d,
				name:         String::new(),
				points_cache: PointsCache::default(),
				clear_cache:  true,
			},
			ui:                                         UiState {
				animating: Arc::new(AtomicBool::new(true)),
				conf,
				scheduled_url_update: false,
				last_url_update: 0.0,
				web_storage,
				next_color,
				plot_bounds: PlotBounds::from_min_max([-2.0, -2.0], [2.0, 2.0]),
				reset_graph: false,

				cur_dir,
				serialization_error,
				stack_overflow_guard: Box::leak(Box::new(AtomicIsize::new(0))),
				eval_errors: AHashMap::default(),
				selected_plot_item: None,
				f32_epsilon: F32NumericType::EPSILON,
				f64_epsilon: DefaultNumericTypes::EPSILON,
				permalink_string: String::new(),
			},
		}
	}

	fn side_panel<T: NumericTypesExt>(
		state: &mut State<T>, ui_state: &mut UiState, ctx: &egui::Context, frame: &mut eframe::Frame,
	) where
		T::Float: Send + Sync + Copy,
		T::Int: Send + Sync, {
		puffin::GlobalProfiler::lock().new_frame();

		puffin::profile_scope!("side_panel");
		SidePanel::left("left_panel").default_width(200.0).show(ctx, |ui| {
			ScrollArea::vertical().show(ui, |ui| {
				ui.add_space(10.0);
				ui.horizontal_top(|ui| {
					ui.heading("Rust Graph");
					if ui.button(if ui_state.conf.dark_mode { "🌙" } else { "☀" }).clicked() {
						ui_state.conf.dark_mode = !ui_state.conf.dark_mode;
					}
				});

				ui.separator();

				let mut needs_recompilation = false;

				ui.horizontal_top(|ui| {
					ui.menu_button("Add", |ui| {
						if ui.button("Function").clicked() {
							state.entries.push(Entry::new_function(ui_state.next_color, "".to_string()));
							ui_state.next_color += 1;
							needs_recompilation = true;
						}
						if ui.button("Constant").clicked() {
							state.entries.push(Entry::new_constant(ui_state.next_color));
							ui_state.next_color += 1;
							needs_recompilation = true;
						}
						if ui.button("Points").clicked() {
							state.entries.push(Entry::new_points(ui_state.next_color));
							ui_state.next_color += 1;
							needs_recompilation = true;
						}
						if ui.button("Integral").clicked() {
							state.entries.push(Entry::new_integral(ui_state.next_color));
							ui_state.next_color += 1;
							// needs_recompilation = true;
						}
					});

					if ui.button("Clear all").clicked() {
						state.entries.clear();
						state.clear_cache = true;
					}
				});

				ui.add_space(4.5);

				let mut remove = None;
				let mut animating = false;
				for (n, entry) in state.entries.iter_mut().enumerate() {
					let (text_col, col) = if entry.visible {
						(Color32::BLACK, entry.color())
					} else {
						(Color32::LIGHT_GRAY, egui::Color32::TRANSPARENT)
					};

					let txt = match &entry.value {
						EntryData::Function { .. } => "λ",
						EntryData::Constant { .. } => {
							if entry.visible {
								"⏸"
							} else {
								"⏵"
							}
						},
						EntryData::Points(_) => "●",
						EntryData::Integral { .. } => "∫",
					};
					ui.with_layout(egui::Layout::right_to_left(Align::LEFT), |ui| {
						if ui.button("X").clicked() {
							remove = Some(n);
							needs_recompilation = true;
						}
						let mut color_picker = MenuButton::new(RichText::new("🎨").color(Color32::BLACK));
						color_picker.button = color_picker.button.fill(entry.color());
						color_picker.ui(ui, |ui| {
							for i in 0..NUM_COLORS {
								if ui.button(RichText::new("     ").background_color(COLORS[i])).clicked() {
									entry.color = i;
								}
							}
						});
						ui.with_layout(egui::Layout::left_to_right(Align::LEFT), |ui| {
							let prev_visible = entry.visible;
							if ui
								.add(
									Button::new(RichText::new(txt).strong().monospace().color(text_col))
										.fill(col)
										.rounding(10),
								)
								.clicked()
							{
								entry.visible = !entry.visible;
							}

							if ui
								.add(
									TextEdit::singleline(&mut entry.name)
										.desired_width(30.0)
										.hint_text("name"),
								)
								.changed()
							{
								needs_recompilation = true;
							}
							let color = entry.color();
							match &mut entry.value {
								EntryData::Function { text, func } => {
									match edit_expr(ui, text, func, "sin(x)", None, state.clear_cache) {
										Ok(changed) => {
											needs_recompilation |= changed;
										},
										Err(e) => {
											ui_state.eval_errors.insert(n, format!("Parsing error: {e}"));
										},
									}
								},
								EntryData::Integral {
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
											match edit_expr(
												ui,
												lower_text,
												lower,
												"lower",
												Some(30.0),
												state.clear_cache,
											) {
												Ok(changed) => {
													// needs_recompilation |= changed;
												},
												Err(e) => {
													ui_state
														.eval_errors
														.insert(n, format!("Parsing error: {e}"));
												},
											}
											ui.label("Upper:");
											match edit_expr(
												ui,
												upper_text,
												upper,
												"upper",
												Some(30.0),
												state.clear_cache,
											) {
												Ok(changed) => {
													// needs_recompilation |= changed;
												},
												Err(e) => {
													ui_state
														.eval_errors
														.insert(n, format!("Parsing error: {e}"));
												},
											}
										});
										ui.horizontal(|ui| {
											ui.label("Func:");
											match edit_expr(
												ui, func_text, func, "func", None, state.clear_cache,
											) {
												Ok(changed) => {
													needs_recompilation |= changed;
												},
												Err(e) => {
													ui_state
														.eval_errors
														.insert(n, format!("Parsing error: {e}"));
												},
											};
											ui.label("dx");
										});
										if let Some(calculated) = calculated {
											ui.label(
												RichText::new(format!("Value: {}", calculated)).color(color),
											);
										}

										ui.add(Slider::new(resolution, 10..=1000).text("Resolution"));
									});
								},
								EntryData::Points(points) => {
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
													state.clear_cache,
												) {
													Ok(changed) => {
														needs_recompilation |= changed;
													},
													Err(e) => {
														ui_state
															.eval_errors
															.insert(n, format!("Parsing error: {e}"));
													},
												}
												match edit_expr(
													ui,
													&mut point.text_y,
													&mut point.y,
													"point_y",
													Some(80.0),
													state.clear_cache,
												) {
													Ok(changed) => {
														needs_recompilation |= changed;
													},
													Err(e) => {
														ui_state
															.eval_errors
															.insert(n, format!("Parsing error: {e}"));
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
												points.push(EntryPoint::default());
											}
										});
									});
								},

								EntryData::Constant { value, step, ty } => {
									let mut v = T::float_to_f64(*value);
									let range = ty.range();
									let start = *range.start();
									let end = *range.end();
									v = v.clamp(start, end);

									ui.vertical(|ui| {
										ui.horizontal(|ui| {
											ui.menu_button(ty.symbol(), |ui| {
												let new_end =
													if end.is_infinite() { start + 10.0 } else { end };
												let lfab = ConstantType::LoopForwardAndBackward {
													start,
													end: new_end,
													forward: true,
												};
												if ui.button(lfab.name()).clicked() {
													*ty = lfab;
													animating = true;
												}
												let lf = ConstantType::LoopForward { start, end: new_end };
												if ui.button(lf.name()).clicked() {
													*ty = lf;
													animating = true;
												}
												let po = ConstantType::PlayOnce { start, end: new_end };
												if ui.button(po.name()).clicked() {
													*ty = po;
													animating = true;
												}
												let pi = ConstantType::PlayIndefinitely { start };
												if ui.button(pi.name()).clicked() {
													*ty = pi;
													animating = true;
												}
											});
											match ty {
												ConstantType::LoopForwardAndBackward {
													start, end, ..
												}
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
												if T::float_to_f64(*value) >= end {
													*value = T::f64_to_float(start);
                          animating = true;
												}
											}

											if entry.visible {
												ctx.request_repaint();
												animating = true;

												match ty {
													ConstantType::LoopForwardAndBackward {
														forward, ..
													} => {
														if T::float_to_f64(*value) > end {
															*forward = false;
														}
														if T::float_to_f64(*value) < start {
															*forward = true;
														}
														if *forward {
															*value =
																T::float_add(*value, T::f64_to_float(*step));
														} else {
															*value =
																T::float_sub(*value, T::f64_to_float(*step));
														}
													},
													ConstantType::LoopForward { .. } => {
														*value = T::float_add(*value, T::f64_to_float(*step));
														if T::float_to_f64(*value) >= end {
															*value = T::f64_to_float(start);
														}
													},
													ConstantType::PlayOnce { .. }
													| ConstantType::PlayIndefinitely { .. } => {
														*value = T::float_add(*value, T::f64_to_float(*step));
														if T::float_to_f64(*value) >= end {
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
											.changed() || state.clear_cache
										{
											*value = T::f64_to_float(v);
											animating = true;
										}
									});
								},
							}
						});
					});
					if let Some(eval_error) = ui_state.eval_errors.get(&n) {
						ui.label(RichText::new(eval_error).color(Color32::RED));
					}
					ui.separator();
				}
				if let Some(n) = remove {
					state.entries.remove(n);
				}
				ui_state.animating.store(animating, Ordering::Relaxed);

				ui.separator();

				CollapsingHeader::new("Settings").default_open(true).show(ui, |ui| {
					ui.horizontal_top(|ui| {
						let mut total_bytes = 0;
						for entry in state.points_cache.iter() {
							total_bytes += 16 * entry.1.capacity()
						}
						ui.label(format!(
							"Points Cache is at least: {}MB",
							total_bytes as f64 / (1024.0 * 1024.0)
						));
						if ui.button("Clear points cache").clicked() {
							state.clear_cache = true;
						}
					});

					ui.separator();
					ui.checkbox(&mut ui_state.conf.use_f32, "Use f32");

					ui.separator();
					ui.add(Slider::new(&mut ui_state.conf.resolution, 10..=1000).text("Resolution"));
				});

				ui.separator();
				#[cfg(target_arch = "wasm32")]
				const PERSISTANCE_TYPE: &str = "Local Storage";
				#[cfg(not(target_arch = "wasm32"))]
				const PERSISTANCE_TYPE: &str = "Persistence";
				CollapsingHeader::new(PERSISTANCE_TYPE).default_open(true).show(ui, |ui| {
					Self::persisence(state, ui_state, ui, frame);
				});

				if needs_recompilation || state.clear_cache {
					puffin::profile_scope!("clear_cache");
					state.points_cache.clear();
					let mut output = Vec::new();
					if serialize_to(&mut output, &state.entries).is_ok() {
						// let mut data_str = String::with_capacity(output.len() + 1);
						let url_encoded = urlencoding::encode(str::from_utf8(&output).unwrap());
						let mut permalink_string = String::with_capacity(url_encoded.len() + 1);
						permalink_string.push_str("#");
						permalink_string.push_str(&url_encoded);
						ui_state.permalink_string = permalink_string;
						ui_state.scheduled_url_update = true;
					}

					state.ctx.write().unwrap().clear();
					state.ctx.write().unwrap().set_builtin_functions_disabled(false).unwrap();
					init_functions::<T>(&mut *(state.ctx.write().unwrap()));
					init_consts::<T>(&mut *(state.ctx.write().unwrap()));

					let main_context = state.ctx;
					let stack_overflow_guard = ui_state.stack_overflow_guard;
					// state.context_stash.lock().unwrap().clear();

					for entry in state.entries.iter_mut() {
						puffin::profile_scope!("entry_process");
						if entry.name != "x" {
							match &mut entry.value {
								EntryData::Points(_) => {},
								EntryData::Integral { .. } => {},
								EntryData::Constant { value, .. } => {
									if !entry.name.is_empty() {
										main_context
											.write()
											.unwrap()
											.set_value(
												entry.name.as_str(),
												evalexpr::Value::<T>::Float(*value),
											)
											.unwrap();
									}
								},
								EntryData::Function { text, func } => {
									if let Some(func) = func.clone() {
										struct LocalCache(Mutex<AHashMap<CacheKey, f64>>);
										impl Clone for LocalCache {
											fn clone(&self) -> Self {
												Self(Mutex::new(self.0.lock().unwrap().clone()))
											}
										}
										impl LocalCache {
											fn lock(&self) -> MutexGuard<'_, AHashMap<CacheKey, f64>> {
												self.0.lock().unwrap()
											}
										}
										let local_cache: LocalCache = LocalCache(Mutex::new(
											AHashMap::with_capacity(ui_state.conf.resolution),
										));
										let animating = ui_state.animating.clone();
										let fun = Function::new(move |v| {
											// puffin::profile_scope!("eval_function");
											if stack_overflow_guard
												.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
												> MAX_FUNCTION_NESTING
											{
												return Err(EvalexprError::CustomMessage(format!(
													"Max function nesting reached ({MAX_FUNCTION_NESTING})"
												)));
											}

											let v = match v {
												Value::Float(x) => *x,
												Value::Boolean(x) => T::f64_to_float(*x as i64 as f64),
												Value::String(_) => T::ZERO,
												Value::Int(x) => T::int_as_float(x),
												Value::Tuple(values) => values[0]
													.as_number()
													.or_else(|_| {
														values[0]
															.as_boolean()
															.map(|x| T::f64_to_float(x as i64 as f64))
													})
													.unwrap_or(T::ZERO),
												Value::Empty => T::ZERO,
											};
											let animating = animating.load(Ordering::Relaxed);
											if !animating {
												if let Some(cached) =
													local_cache.lock().get(&CacheKey(T::float_to_f64(v)))
												{
													stack_overflow_guard
														.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
													return Ok(Value::Float(T::f64_to_float(*cached)));
												}
											}
											let vv = Value::<T>::Float(v);

											let context = main_context.read().unwrap();

											let res = { func.eval_number_with_context_and_x(&*context, &vv) };
											stack_overflow_guard
												.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
											res.map(|res| {
												if !animating {
													local_cache.lock().insert(
														CacheKey(T::float_to_f64(v)),
														T::float_to_f64(res),
													);
												}
												Value::Float(res)
											})
										});

										let name = if entry.name.is_empty() {
											text.clone()
										} else {
											entry.name.clone()
										};

										main_context.write().unwrap().set_function(name, fun).unwrap();
									}
								},
							}
						}
					}
				} else if animating {
					let main_context = state.ctx;
					for entry in state.entries.iter_mut() {
						puffin::profile_scope!("entry_process");
						if entry.name != "x" {
							match &mut entry.value {
								EntryData::Constant { value, .. } => {
									if !entry.name.is_empty() {
										main_context
											.write()
											.unwrap()
											.set_value(
												entry.name.as_str(),
												evalexpr::Value::<T>::Float(*value),
											)
											.unwrap();
									}
								},
								_ => {},
							}
						}
					}
				}

				if ui_state.scheduled_url_update {
					let time = ui.input(|i| i.time);

					if time - 1.0 > ui_state.last_url_update {
						ui_state.last_url_update = time;
						ui_state.scheduled_url_update = false;

						#[cfg(target_arch = "wasm32")]
						{
							use eframe::wasm_bindgen::prelude::*;
							let history = web_sys::window()
								.expect("Couldn't get window")
								.history()
								.expect("Couldn't get window.history");
							history
								.push_state_with_url(&JsValue::NULL, "", Some(&ui_state.permalink_string))
								.unwrap();
						}
					}
				}
				state.clear_cache = false;
				if let Some(error) = &ui_state.serialization_error {
					ui.label(RichText::new(error).color(Color32::RED));
				}

				ui.separator();
				ui.hyperlink_to("Github", "https://github.com/vdrn/rust_graph");
				#[cfg(not(target_arch = "wasm32"))]
				ui.hyperlink_to("View Online", {
					let mut base_url = "https://vdrn.github.io/rust_graph/".to_string();
					base_url.push_str(ui_state.permalink_string.as_str());

					base_url
				});
			});
		});

		ui_state.eval_errors.clear();
	}
	fn save_file<T: NumericTypesExt>(ui_state: &mut UiState, state: &State<T>, frame: &mut eframe::Frame)
	where
		T::Float: Send + Sync + Copy,
		T::Int: Send + Sync, {
		#[cfg(target_arch = "wasm32")]
		{
			let file = format!("{}.json", state.name);
			let mut output = Vec::new();
			if let Err(e) = serialize_to(&mut output, &state.entries) {
				ui_state.serialization_error = Some(e.to_string());
			} else {
				ui_state.serialization_error = None;
				ui_state.web_storage.insert(file, String::from_utf8(output).unwrap());
				if let Some(storage) = frame.storage_mut() {
					storage.flush();
				}
			}
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			let save_path = PathBuf::from(&ui_state.cur_dir).join(format!("{}.json", state.name));
			if let Some(parent) = save_path.parent() {
				// Recursively create all parent directories if they don't exist
				fs::create_dir_all(parent).ok();
			}
			let Ok(mut file) = std::fs::File::create(&save_path) else {
				ui_state.serialization_error = Some(format!("Could not create file: {}", save_path.display()));
				return;
			};
			if let Err(e) = serialize_to(&mut file, &state.entries) {
				ui_state.serialization_error = Some(e.to_string());
			} else {
				ui_state.serialization_error = None;
				Self::load_file_entries(&ui_state.cur_dir, &mut ui_state.web_storage);
			}
		}
	}
	fn load_file_entries(cur_dir: &str, web_st: &mut AHashMap<String, String>) {
		web_st.clear();
		let Ok(entries) = std::fs::read_dir(PathBuf::from(cur_dir)) else {
			// ui.label("No entries found");
			return;
		};
		for entry in entries {
			let Ok(entry) = entry else {
				continue;
			};
			let file_name = entry.file_name();
			let Some(file_name) = file_name.to_str() else {
				continue;
			};
			if !file_name.ends_with(".json") {
				continue;
			}

			web_st.insert(file_name.to_string(), String::new());
		}
	}
	fn load_file<T: NumericTypesExt>(
		cur_dir: &str, web_st: &AHashMap<String, String>, file_name: &str, state: &mut State<T>,
	) -> Result<(), String>
	where
		T::Float: Send + Sync + Copy,
		T::Int: Send + Sync, {
		#[cfg(target_arch = "wasm32")]
		{
			if let Some(file) = web_st.get(file_name) {
				let entries = deserialize_from::<T>(file.as_bytes()).unwrap();
				state.entries = entries;
				state.name = file_name.strip_suffix(".json").unwrap().to_string();
			}
		}

		#[cfg(not(target_arch = "wasm32"))]
		{
			let Ok(mut file) = std::fs::read(PathBuf::from(cur_dir).join(file_name)) else {
				return Err(format!("Could not open file: {}", file_name));
			};
			let entries = deserialize_from::<T>(&mut file).unwrap();
			state.entries = entries;
			state.name = file_name.strip_suffix(".json").unwrap().to_string();
		}

		state.clear_cache = true;
		Ok(())
	}
	fn persisence<T: NumericTypesExt>(
		state: &mut State<T>, ui_state: &mut UiState, ui: &mut egui::Ui, frame: &mut eframe::Frame,
	) where
		T::Float: Send + Sync + Copy,
		T::Int: Send + Sync, {
		ui.horizontal_top(|ui| {
			ui.label("Name:");
			ui.text_edit_singleline(&mut state.name);
			if !state.name.trim().is_empty() && ui.button("Save").clicked() {
				Self::save_file(ui_state, state, frame);
			}
		});

		#[cfg(not(target_arch = "wasm32"))]
		{
			ui.separator();
			ui.horizontal_top(|ui| {
				ui.label("CWD:");
				let changed = ui.text_edit_singleline(&mut ui_state.cur_dir).changed();

				if ui.button("⟳").clicked() || changed {
					Self::load_file_entries(&ui_state.cur_dir, &mut ui_state.web_storage);
				}
			});
		}

		if !ui_state.web_storage.is_empty() {
			ui.separator();
		}
		ui.horizontal(|ui| {
			Grid::new("files").num_columns(2).striped(true).show(ui, |ui| {
				for file_name in ui_state.web_storage.keys() {
					ui.label(file_name);
					if ui.button("Load").clicked() {
						if let Err(e) =
							Self::load_file(&ui_state.cur_dir, &ui_state.web_storage, &file_name, state)
						{
							ui_state.serialization_error = Some(format!("Could not open file: {}", e));
							return;
						};
					}
					ui.end_row();
				}
			});
		});
	}

	fn graph_panel<T: NumericTypesExt>(state: &mut State<T>, ui_state: &mut UiState, ctx: &egui::Context)
	where
		T::Float: Send + Sync + Copy,
		T::Int: Send + Sync + Copy, {
		let main_context = state.ctx;
		let stack_overflow_guard = ui_state.stack_overflow_guard;
		let first_x = ui_state.plot_bounds.min()[0];
		let last_x = ui_state.plot_bounds.max()[0];
		let (first_x, last_x) = snap_range_to_grid(first_x, last_x, 10.0);
		let plot_width = last_x - first_x;
		let mut points_to_draw = ui_state.conf.resolution.max(1);
		let mut step_size = plot_width / points_to_draw as f64;
		while points_to_draw > 2 && first_x + step_size == first_x {
			points_to_draw /= 2;
			step_size = plot_width / points_to_draw as f64;
		}

		let animating = ui_state.animating.load(Ordering::Relaxed);
		puffin::profile_scope!("graph");

		let eps = if ui_state.conf.use_f32 { ui_state.f32_epsilon } else { ui_state.f64_epsilon };

		let mut lines = vec![];
		let mut points = vec![];
		let mut polys = vec![];

		'next_entry: for (ei, entry) in state.entries.iter_mut().enumerate() {
			let visible = entry.visible;
			if !visible && !matches!(entry.value, EntryData::Integral { .. }) {
				continue;
			}
			puffin::profile_scope!("entry_draw", entry.name.clone());
			let id = Id::new(ei);
			stack_overflow_guard.store(0, std::sync::atomic::Ordering::Relaxed);
			let color = entry.color();

			match &mut entry.value {
				EntryData::Constant { .. } => {},
				EntryData::Integral { func, func_text, lower, upper, calculated, resolution, .. } => {
					match (lower, upper, func) {
						(Some(lower), Some(upper), Some(func)) => {
							let lower = match lower.eval_number_with_context(&*main_context.read().unwrap()) {
								Ok(lower) => T::float_to_f64(lower),
								Err(e) => {
									ui_state
										.eval_errors
										.insert(ei, format!("Error evaluating lower bound: {e}"));
									continue 'next_entry;
								},
							};
							let upper = match upper.eval_number_with_context(&*main_context.read().unwrap()) {
								Ok(upper) => T::float_to_f64(upper),
								Err(e) => {
									ui_state
										.eval_errors
										.insert(ei, format!("Error evaluating upper bound: {e}"));
									continue 'next_entry;
								},
							};
							let range = upper - lower;
							if lower > upper {
								ui_state
									.eval_errors
									.insert(ei, "Lower bound must be less than upper bound".to_string());
								*calculated = None;

								continue 'next_entry;
							}
							let resolution = *resolution;
							let step = (range / resolution as f64);
							if lower + step == lower {
								*calculated = Some(T::ZERO);
								continue 'next_entry;
							}
							*calculated = None;

							let mut cache = (!animating).then(|| {
								state
									.points_cache
									.entry(func_text.clone())
									.or_insert_with(|| AHashMap::with_capacity(ui_state.conf.resolution))
							});

							let mut int_lines = vec![];
							let mut fun_lines = vec![];
							let stroke_color = color;
							let rgba_color = stroke_color.to_srgba_unmultiplied();
							let fill_color = Color32::from_rgba_unmultiplied(
								rgba_color[0], rgba_color[1], rgba_color[1], 128,
							);

							let mut x = lower;
							let mut result: T::Float = T::ZERO;
							let mut prev_y = None;
							for i in 0..(resolution + 1) {
								let x = lower + step * i as f64;

								let xx = evalexpr::Value::<T>::Float(T::f64_to_float(x));

								let y_f64 = if let Some(cache) = &mut cache {
									let val = cache.entry(CacheKey(x));
									match val {
										hash_map::Entry::Occupied(entry) => entry.get().clone(),
										hash_map::Entry::Vacant(entry) => {
											match func.eval_number_with_context_and_x(
												&*main_context.read().unwrap(),
												&xx,
											) {
												Ok(y) => {
													let y = T::float_to_f64(y);
													entry.insert(y);
													y
												},
												Err(e) => {
													ui_state.eval_errors.insert(ei, e.to_string());
													continue 'next_entry;
												},
											}
										},
									}
								} else {
									match func
										.eval_number_with_context_and_x(&*main_context.read().unwrap(), &xx)
									{
										Ok(y) => T::float_to_f64(y),
										Err(e) => {
											ui_state.eval_errors.insert(ei, e.to_string());
											continue 'next_entry;
										},
									}
								};

								let y = T::f64_to_float(y_f64);
								if let Some(prev_y) = prev_y {
									let eps = 0.0;
									let prev_y_f64 = T::float_to_f64(prev_y);

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
											polys.push(triangle1);
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
											polys.push(triangle2);
										}

										let t = T::f64_to_float(t);
										let step = T::f64_to_float(step);

										let step1 = T::float_mul(step, t);
										let step2 = T::float_sub(step, step1);

										let b1 = T::float_mul(prev_y, step1);
										let b2 = T::float_mul(y, step2);
										result = T::float_add(result, T::float_mul(b1, T::HALF));
										result = T::float_add(result, T::float_mul(b2, T::HALF));
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
											polys.push(poly);
										}
										let dy = T::float_sub(y, prev_y);
										let step = T::f64_to_float(step);
										let d = T::float_mul(dy, step);
										result = T::float_add(result, T::float_mul(prev_y, step));
										result = T::float_add(result, T::float_mul(d, T::HALF));
									}
								}
								if T::float_to_f64(result).is_nan() {
									ui_state.eval_errors.insert(ei, "Integral is undefined".to_string());
									continue 'next_entry;
								}

								if visible {
									int_lines.push(PlotPoint::new(x, T::float_to_f64(result)));
									fun_lines.push(PlotPoint::new(x, y_f64));
								}
								prev_y = Some(y);
								// x += step;
							}
							let int_name =
								if entry.name.is_empty() { func_text.as_str() } else { entry.name.as_str() };
							lines.push((
								false,
								Id::NULL,
								Line::new(
									format!("∫[{},x]({})dx", lower, int_name),
									PlotPoints::Owned(int_lines),
								)
								.color(stroke_color),
							));
							lines.push((
								false,
								Id::NULL,
								Line::new("", PlotPoints::Owned(fun_lines)).color(stroke_color),
							));
							*calculated = Some(result);
						},
						_ => {},
					}
				},
				EntryData::Points(ps) => {
					main_context
						.write()
						.unwrap()
						.set_value("x", evalexpr::Value::<T>::Float(T::ZERO))
						.unwrap();
					let mut line_buffer = vec![];
					for p in ps {
						match eval_point(&*main_context.read().unwrap(), p) {
							Ok(Some((x, y))) => line_buffer.push([x, y]),
							Err(e) => {
								ui_state.eval_errors.insert(ei, e);
								continue 'next_entry;
							},
							_ => {},
						}
					}

					if line_buffer.len() == 1 {
						let radius = if Some(id) == ui_state.selected_plot_item { 5.0 } else { 3.5 };
						points.push(
							Points::new(entry.name.clone(), line_buffer[0]).id(id).color(color).radius(radius),
						);
					} else if line_buffer.len() > 1 {
						let line = Line::new(entry.name.clone(), line_buffer)
							.color(color)
							.id(id)
							.width(if Some(id) == ui_state.selected_plot_item { 3.5 } else { 1.0 });
						lines.push((false, id, line));
						// plot_ui.line(line);
					}
				},
				EntryData::Function { text, func } => {
					if let Some(func) = func {
						let name = if entry.name.is_empty() {
							format!("y = {}", text)
						} else {
							format!("{}(x) = {}", entry.name, text)
						};
						let mut cache = (!animating).then(|| {
							state
								.points_cache
								.entry(text.clone())
								.or_insert_with(|| AHashMap::with_capacity(ui_state.conf.resolution))
						});

						let mut local_optima_buffer = vec![];
						let mut pp_buffer = vec![];

						let mut x = first_x;
						let mut prev: [Option<(f64, f64)>; 2] = [None; 2];
						while x < last_x {
							let cur_x = x;
							puffin::profile_scope!("graph_step");

							let cur_y = if let Some(cache) = &mut cache {
								let value = cache.entry(CacheKey(x));

								match value {
									hash_map::Entry::Occupied(entry) => entry.get().clone(),
									hash_map::Entry::Vacant(entry) => {
										let x = evalexpr::Value::<T>::Float(T::f64_to_float(x));
										match func
											.eval_number_with_context_and_x(&*main_context.read().unwrap(), &x)
										{
											Ok(y) => {
												entry.insert(T::float_to_f64(y));
												T::float_to_f64(y)
											},
											Err(e) => {
												ui_state.eval_errors.insert(ei, e.to_string());

												continue 'next_entry;
											},
										}
									},
								}
							} else {
								let x = evalexpr::Value::<T>::Float(T::f64_to_float(x));
								match func.eval_number_with_context_and_x(&*main_context.read().unwrap(), &x) {
									Ok(y) => T::float_to_f64(y),
									Err(e) => {
										ui_state.eval_errors.insert(ei, e.to_string());

										continue 'next_entry;
									},
								}
							};
							pp_buffer.push(PlotPoint::new(x, cur_y));
							let cur = (cur_x, cur_y);
							fn less_then(a: f64, b: f64, e: f64) -> bool { b - a > e }
							fn greater_then(a: f64, b: f64, e: f64) -> bool { a - b > e }
							if Some(id) == ui_state.selected_plot_item {
								if let (Some(prev_0), Some(prev_1)) = (prev[0], prev[1]) {
									if less_then(prev_0.1, prev_1.1, eps) && greater_then(prev_1.1, cur.1, eps)
									{
										local_optima_buffer.push(
											Points::new("", [prev_1.0, prev_1.1])
												.color(Color32::GRAY)
												.radius(3.5),
										);
									}
									if greater_then(prev_0.1, prev_1.1, eps) && less_then(prev_1.1, cur.1, eps)
									{
										local_optima_buffer.push(
											Points::new("", [prev_1.0, prev_1.1])
												.color(Color32::GRAY)
												.radius(3.5),
										);
									}
								}
								prev[0] = prev[1];
								prev[1] = Some(cur);
							}

							let prev_x = x;
							x += step_size;
							if x == prev_x {
								break;
							}
						}

						let line = Line::new(name.clone(), PlotPoints::Owned(pp_buffer))
							.id(id)
							.width(if Some(id) == ui_state.selected_plot_item { 3.5 } else { 1.0 })
							.color(entry.color());
						lines.push((true, id, line));

						for point in local_optima_buffer {
							points.push(point);
						}
					}
				},
			}
		}

		egui::CentralPanel::default().show(ctx, |ui| {
			let plot_id = ui.make_persistent_id("Plot");
			let mut plot = Plot::new(plot_id)
				.id(plot_id)
				.legend(Legend::default().text_style(TextStyle::Body))
				.default_x_bounds(ui_state.plot_bounds.min()[0], ui_state.plot_bounds.max()[0])
				.default_y_bounds(ui_state.plot_bounds.min()[1], ui_state.plot_bounds.max()[1]);

			if ui_state.reset_graph {
				ui_state.reset_graph = false;
				plot = plot.data_aspect(1.0).center_x_axis(true).center_y_axis(true).reset();
			}

			let plot_res = plot.show(ui, |plot_ui| {
				puffin::profile_scope!("graph_show");
				plot_ui.hline(HLine::new("", 0.0).color(Color32::WHITE));
				plot_ui.vline(VLine::new("", 0.0).color(Color32::WHITE));
				for (implicit, line_id, line) in lines.iter() {
					if !*implicit || Some(*line_id) != ui_state.selected_plot_item {
						continue;
					}
					let PlotGeometry::Points(plot_points) = line.geometry() else {
						continue;
					};
					for (p_i, point) in plot_points.iter().enumerate() {
						let Some(point_before) = plot_points.get(p_i - 1) else {
							continue;
						};
						for (other_implicit, other_line_id, other_line) in lines.iter() {
							if !other_implicit || other_line_id == line_id {
								continue;
							}
							let PlotGeometry::Points(other_points) = other_line.geometry() else {
								continue;
							};
							let (Some(other_point_before), Some(other_point)) =
								(other_points.get(p_i - 1), other_points.get(p_i))
							else {
								continue;
							};
							if let Some((t, y)) = implicit_intersects(
								point_before.y, point.y, other_point_before.y, other_point.y,
							) {
								points.push(
									Points::new("", [point_before.x + (point.x - point_before.x) * t, y])
										.color(Color32::GRAY)
										.radius(3.5),
								);
							}
						}
					}
				}
				puffin::profile_scope!("graph_lines");
				for poly in polys {
					plot_ui.polygon(poly);
				}
				for (_, _, line) in lines {
					plot_ui.line(line);
				}
				for point in points {
					plot_ui.points(point);
				}
			});

			if plot_res.response.double_clicked() {
				ui_state.reset_graph = true;
			}

			ui_state.plot_bounds = plot_res.transform.bounds().clone();
			if let Some(id) = plot_res.hovered_plot_item {
				if ui.input(|i| i.pointer.primary_clicked()) {
					ui_state.selected_plot_item = Some(id);
				}
			} else {
				if ui.input(|i| i.pointer.primary_clicked()) {
					ui_state.selected_plot_item = None;
				}
			}
		});
	}
}
fn edit_expr<T: NumericTypesExt>(
	ui: &mut egui::Ui, text: &mut String, expr: &mut Option<Node<T>>, hint_text: &str,
	desired_width: Option<f32>, force_update: bool,
) -> Result<bool, String>
where
	T::Float: Send + Sync,
	T::Int: Send + Sync, {
	let mut text_edit = TextEdit::singleline(text).hint_text(hint_text);
	if let Some(width) = desired_width {
		text_edit = text_edit.desired_width(width);
	}
	if ui.add(text_edit).changed() || force_update {
		if text != "" {
			*expr = match evalexpr::build_operator_tree::<T>(text) {
				Ok(func) => Some(func),
				Err(e) => {
					return Err(e.to_string());
				},
			};
		} else {
			*expr = None;
		}

		return Ok(true);
	}
	Ok(false)
}
fn eval_point<T: NumericTypesExt>(
	ctx: &evalexpr::HashMapContext<T>, p: &EntryPoint<T>,
) -> Result<Option<(f64, f64)>, String>
where
	T::Float: Send + Sync,
	T::Int: Send + Sync, {
	match (&p.x, &p.y) {
		(Some(x), Some(y)) => {
			let x = x.eval_number_with_context(ctx).map_err(|e| e.to_string())?;
			let y = y.eval_number_with_context(ctx).map_err(|e| e.to_string())?;
			return Ok(Some((T::float_to_f64(x), T::float_to_f64(y))));
		},
		_ => {},
	}
	Ok(None)
}
fn implicit_intersects(p1_y: f64, p2_y: f64, o_p1_y: f64, o_p2_y: f64) -> Option<(f64, f64)> {
	let m1 = p2_y - p1_y;
	let m2 = o_p2_y - o_p1_y;

	// Check if lines are parallel
	let slope_diff = m1 - m2;
	if slope_diff.abs() < 1e-10 {
		return None;
	}

	// Line equations:
	// y1 = p1_y + m1 * x
	// y2 = o_p1_y + m2 * x

	let x = (o_p1_y - p1_y) / slope_diff;

	if x < 0.0 || x > 1.0 {
		return None;
	}

	let y = p1_y + m1 * x;

	Some((x, y))
}

impl App for Application {
	fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
		if self.ui.conf.dark_mode {
			ctx.set_visuals(Visuals::dark());
		} else {
			ctx.set_visuals(Visuals::light());
		}
		ctx.set_pixels_per_point(1.5);

		let use_f32 = self.ui.conf.use_f32;
		if use_f32 {
			Self::side_panel(&mut self.s, &mut self.ui, ctx, frame);
			Self::graph_panel(&mut self.s, &mut self.ui, ctx);
		} else {
			Self::side_panel(&mut self.d, &mut self.ui, ctx, frame);
			Self::graph_panel(&mut self.d, &mut self.ui, ctx);
		}
		if use_f32 != self.ui.conf.use_f32 {
			if use_f32 {
				let mut output = Vec::with_capacity(1024);
				if serialize_to(&mut output, &self.s.entries).is_ok() {
					self.d.entries = deserialize_from(&output).unwrap();
					self.d.clear_cache = true;
					self.d.name = self.s.name.clone();
				}
			} else {
				let mut output = Vec::with_capacity(1024);
				if serialize_to(&mut output, &self.d.entries).is_ok() {
					self.s.entries = deserialize_from(&output).unwrap();
					self.s.clear_cache = true;
					self.s.name = self.d.name.clone();
				}
			}
		}
		if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
			ctx.send_viewport_cmd(egui::ViewportCommand::Close);
		}
	}

	fn save(&mut self, storage: &mut dyn eframe::Storage) {
		storage.set_string(DATA_KEY, serde_json::to_string(&self.ui.web_storage).unwrap());
		storage.set_string(CONF_KEY, serde_json::to_string(&self.ui.conf).unwrap());
	}

	fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {}

	fn auto_save_interval(&self) -> std::time::Duration { std::time::Duration::from_secs(5) }

	fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
		egui::Color32::from_rgba_unmultiplied(12, 12, 12, 180).to_normalized_gamma_f32()
	}

	fn persist_egui_memory(&self) -> bool { true }

	fn raw_input_hook(&mut self, _ctx: &egui::Context, _raw_input: &mut egui::RawInput) {}
}

#[derive(Clone, Debug)]
pub struct Entry<T: NumericTypesExt>
where
	T::Float: Send + Sync,
	T::Int: Send + Sync, {
	name:    String,
	visible: bool,
	color:   usize,
	value:   EntryData<T>,
}
#[derive(Clone, Debug)]
struct EntryPoint<T: NumericTypesExt>
where
	T::Float: Send + Sync,
	T::Int: Send + Sync, {
	text_x: String,
	x:      Option<Node<T>>,
	text_y: String,
	y:      Option<Node<T>>,
}

impl<T: NumericTypesExt> Default for EntryPoint<T>
where
	T::Float: Send + Sync,
	T::Int: Send + Sync,
{
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
	fn range(&self) -> RangeInclusive<f64> {
		match self {
			ConstantType::LoopForwardAndBackward { start, end, .. } => (*start..=*end).into(),
			ConstantType::LoopForward { start, end } => (*start..=*end).into(),
			ConstantType::PlayOnce { start, end } => (*start..=*end).into(),
			ConstantType::PlayIndefinitely { start } => (*start..=f64::INFINITY).into(),
		}
	}
	fn symbol(&self) -> &'static str {
		match self {
			ConstantType::LoopForwardAndBackward { .. } => "🔁",
			ConstantType::LoopForward { .. } => "🔂",
			ConstantType::PlayOnce { .. } => "⏯",
			ConstantType::PlayIndefinitely { .. } => "🔀",
		}
	}
	fn name(&self) -> &'static str {
		match self {
			ConstantType::LoopForwardAndBackward { .. } => "🔁 Loop forward and Backward",
			ConstantType::LoopForward { .. } => "🔂 Loop forward",
			ConstantType::PlayOnce { .. } => "⏯ Play once",
			ConstantType::PlayIndefinitely { .. } => "🔀 Play indefinitely",
		}
	}
}
#[derive(Clone, Debug)]
enum EntryData<T: NumericTypesExt>
where
	T::Float: Send + Sync,
	T::Int: Send + Sync, {
	Function {
		text: String,
		func: Option<Node<T>>,
	},
	Constant {
		value: T::Float,
		step:  f64,
		ty:    ConstantType,
	},
	Points(Vec<EntryPoint<T>>),
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
#[derive(Clone, Debug, PartialEq, Copy)]
pub enum EntryType {
	Function,
	Constant,
	Points,
	Integral,
}

impl<T: NumericTypesExt> Entry<T>
where
	T::Float: Send + Sync,
	T::Int: Send + Sync,
{
	pub fn color(&self) -> Color32 { COLORS[self.color % NUM_COLORS] }
	pub fn new_function(color: usize, text: String) -> Self {
		Self {
			color:   color % NUM_COLORS,
			visible: true,
			name:    String::new(),
			value:   EntryData::Function { text, func: None },
		}
	}
	pub fn new_constant(color: usize) -> Self {
		Self {
			color:   color % NUM_COLORS,
			visible: false,
			name:    String::new(),
			value:   EntryData::Constant {
				value: T::ZERO,
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
			value:   EntryData::Points(vec![EntryPoint::default()]),
		}
	}
	pub fn new_integral(color: usize) -> Self {
		Self {
			color:   color % NUM_COLORS,
			visible: true,
			name:    String::new(),
			value:   EntryData::Integral {
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

#[derive(Serialize, Deserialize)]
pub struct EntrySerialized {
	name:    String,
	visible: bool,
	color:   usize,
	value:   EntryValueSerialized,
}
#[derive(Serialize, Deserialize)]
pub enum EntryValueSerialized {
	Function(String),
	Constant {
		value: f64,
		step:  f64,
		ty:    ConstantType,
	},
	Points(Vec<EntryPointSerialized>),
	Integral {
		func_text:  String,
		lower_text: String,
		upper_text: String,

		#[serde(default)]
		resolution: usize,
	},
}
#[derive(Serialize, Deserialize)]
pub struct EntryPointSerialized {
	x: String,
	y: String,
}

pub fn serialize_to<T: NumericTypesExt>(writer: impl Write, entries: &[Entry<T>]) -> std::io::Result<()>
where
	T::Float: Send + Sync + Copy,
	T::Int: Send + Sync, {
	let mut result = Vec::new();
	for entry in entries {
		let entry_serialized = EntrySerialized {
			name:    entry.name.clone(),
			visible: entry.visible,
			color:   entry.color,
			value:   match &entry.value {
				EntryData::Function { text, .. } => EntryValueSerialized::Function(text.clone()),
				EntryData::Constant { value, step, ty } => EntryValueSerialized::Constant {
					value: T::float_to_f64(*value),
					step:  *step,
					ty:    ty.clone(),
				},
				EntryData::Points(points) => {
					let mut points_serialized = Vec::new();
					for point in points {
						let point_serialized =
							EntryPointSerialized { x: point.text_x.clone(), y: point.text_y.clone() };
						points_serialized.push(point_serialized);
					}
					EntryValueSerialized::Points(points_serialized)
				},
				EntryData::Integral { func_text, lower_text, upper_text, resolution, .. } => {
					EntryValueSerialized::Integral {
						func_text:  func_text.clone(),
						lower_text: lower_text.clone(),
						upper_text: upper_text.clone(),
						resolution: *resolution,
					}
				},
			},
		};
		result.push(entry_serialized);
	}
	serde_json::to_writer(writer, &result)?;
	Ok(())
}

#[cfg(target_arch = "wasm32")]
pub fn deserialize_from_url<T: NumericTypesExt>() -> Result<Vec<Entry<T>>, String>
where
	T::Float: Send + Sync + Copy,
	T::Int: Send + Sync, {
	let href = web_sys::window()
		.expect("Couldn't get window")
		.document()
		.expect("Couldn't get document")
		.location()
		.expect("Couldn't get location")
		.href()
		.expect("Couldn't get href");

	if !href.contains('#') {
		return Ok(Vec::new());
	}
	let Some(without_prefix) = href.split('#').last() else {
		return Ok(Vec::new());
	};

	let decoded = urlencoding::decode(without_prefix).map_err(|e| e.to_string())?;
	deserialize_from(decoded.as_bytes())
}

pub fn deserialize_from<T: NumericTypesExt>(reader: &[u8]) -> Result<Vec<Entry<T>>, String>
where
	T::Float: Send + Sync + Copy,
	T::Int: Send + Sync, {
	let entries: Vec<EntrySerialized> = serde_json::from_slice(reader).map_err(|e| e.to_string())?;
	let mut result = Vec::new();
	for entry in entries {
		let entry_deserialized = Entry {
			name:    entry.name,
			visible: entry.visible,
			color:   entry.color,
			value:   match entry.value {
				EntryValueSerialized::Function(text) => {
					EntryData::Function { func: evalexpr::build_operator_tree::<T>(&text).ok(), text }
				},
				EntryValueSerialized::Constant { value, step, ty } => {
					EntryData::Constant { value: T::f64_to_float(value), step, ty }
				},
				EntryValueSerialized::Points(points) => {
					let mut points_deserialized = Vec::new();
					for point in points {
						let point_deserialized = EntryPoint {
							x:      evalexpr::build_operator_tree::<T>(&point.x).ok(),
							y:      evalexpr::build_operator_tree::<T>(&point.y).ok(),
							text_x: point.x,
							text_y: point.y,
						};
						points_deserialized.push(point_deserialized);
					}
					EntryData::Points(points_deserialized)
				},
				EntryValueSerialized::Integral { func_text, lower_text, upper_text, resolution } => {
					EntryData::Integral {
						func: evalexpr::build_operator_tree::<T>(&func_text).ok(),
						lower: evalexpr::build_operator_tree::<T>(&lower_text).ok(),
						upper: evalexpr::build_operator_tree::<T>(&upper_text).ok(),
						func_text,
						lower_text,
						upper_text,
						calculated: None,
						resolution: resolution.max(10),
					}
				},
			},
		};
		result.push(entry_deserialized);
	}
	Ok(result)
}

pub fn snap_range_to_grid(start: f64, end: f64, expand_percent: f64) -> (f64, f64) {
	let range = end - start;
	let expansion = range * (expand_percent / 100.0);

	// Expand the range
	let expanded_start = start - expansion;
	let expanded_end = end + expansion;
	let expanded_range = expanded_end - expanded_start;

	let grid_size = calculate_grid_size_dynamic(expanded_range);

	// snap to boundaries
	let snapped_start = (expanded_start / grid_size).floor() * grid_size;
	let snapped_end = (expanded_end / grid_size).ceil() * grid_size;

	(snapped_start, snapped_end)
}

/// Calculate grid size dynamically with hysteresis
fn calculate_grid_size_dynamic(range: f64) -> f64 {
	let range = range.abs();
	if range == 0.0 {
		return 1.0;
	}

	let exp = range.log10().floor();
	let power = 10_f64.powf(exp);
	let normalized = range / power;

	// Use wider buckets to reduce threshold sensitivity
	let multiplier = if normalized < 2.5 {
		0.5
	} else if normalized < 5.0 {
		1.0
	} else if normalized < 7.5 {
		2.0
	} else {
		5.0
	};
	multiplier * power
}
