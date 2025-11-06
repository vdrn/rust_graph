extern crate alloc;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicIsize, Ordering};
use std::env;
use std::sync::RwLock;

use ahash::AHashMap;
use eframe::egui::{self, CollapsingHeader, Id, RichText, ScrollArea, SidePanel, Slider, TextStyle, Visuals};
use eframe::epaint::Color32;
use eframe::{App, Storage};
use egui_plot::{HLine, Legend, Plot, PlotBounds, PlotGeometry, PlotItem, Points, VLine};
use evalexpr::{
	Context, ContextWithMutableFunctions, ContextWithMutableVariables, DefaultNumericTypes, EvalexprFloat, EvalexprNumericTypes, F32NumericTypes, Value
};
use serde::{Deserialize, Serialize};

mod entry;
mod persistence;
use crate::entry::{ConstantType, Entry, EntryType, PointEntry};

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

struct State<T: EvalexprNumericTypes> {
	entries:     Vec<Entry<T>>,
	ctx:         Arc<RwLock<evalexpr::HashMapContext<T>>>,
	// context_stash: &'static Mutex<Vec<evalexpr::HashMapContext<T>>>,
	name:        String,
	// points_cache: PointsCache,
	clear_cache: bool,
}
fn init_consts<T: EvalexprNumericTypes>(ctx: &mut evalexpr::HashMapContext<T>) {
	macro_rules! add_const {
		($ident:ident, $uppercase:expr, $lowercase:expr) => {
			ctx.set_value($lowercase, Value::Float(T::Float::f64_to_float(core::f64::consts::$ident)))
				.unwrap();
			ctx.set_value($uppercase, Value::Float(T::Float::f64_to_float(core::f64::consts::$ident)))
				.unwrap();
		};
	}

	add_const!(E, "E", "e");
	add_const!(PI, "PI", "pi");
	add_const!(TAU, "TAU", "tau");
}
fn init_functions<T: EvalexprNumericTypes>(ctx: &mut evalexpr::HashMapContext<T>) {
	macro_rules! add_function {
		($ident:ident) => {
			ctx.set_function(
				stringify!($ident).to_string(),
				evalexpr::Function::new(move |v| {
					let v: T::Float = v.as_float()?;
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
					let v2: T::Float = v.pop().unwrap().as_float()?;
					let v1: T::Float = v.pop().unwrap().as_float()?;

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

	ui_scale: f32,
}
impl Default for AppConfig {
	fn default() -> Self { Self { dark_mode: true, use_f32: false, resolution: 500, ui_scale: 1.5 } }
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
	stack_overflow_guard: Arc<AtomicIsize>,
	eval_errors:          AHashMap<usize, String>,
	selected_plot_item:   Option<Id>,

	f32_epsilon:          f64,
	f64_epsilon:          f64,
	permalink_string:     String,
	scheduled_url_update: bool,
	last_url_update:      f64,
	// animating:            Arc<AtomicBool>,
	file_to_remove:       Option<String>,

	lines:    Vec<(bool, Id, egui_plot::Line<'static>)>,
	points:   Vec<egui_plot::Points<'static>>,
	polygons: Vec<egui_plot::Polygon<'static>>,
}
// #[derive(Clone, Debug)]
pub struct Application {
	state_f64: State<DefaultNumericTypes>,
	state_f32: State<F32NumericTypes>,
	ui:        UiState,
	#[cfg(not(target_arch = "wasm32"))]
	_puffin:   puffin_http::Server,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_puffin_server() -> puffin_http::Server {
	let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
	let puffin_server = puffin_http::Server::new(&server_addr).unwrap();
	println!("Run this to view profiling data:  puffin_viewer {server_addr}");

	puffin::set_scopes_on(false);

	puffin_server
}
impl Application {
	pub fn new(storage: Option<&dyn Storage>) -> Self {
		let mut entries_s = Vec::new();
		let mut entries_d = Vec::new();

		// TODO
		let mut web_storage = AHashMap::new();

		#[allow(unused_mut)]
		let mut serialization_error = None;

		let conf = storage
			.and_then(|s| s.get_string(CONF_KEY).and_then(|d| serde_json::from_str(&d).ok()))
			.unwrap_or_default();

		cfg_if::cfg_if! {
			  if #[cfg(target_arch = "wasm32")] {
			let cur_dir = String::new();
			// 	let error: Option<String> = web::get_data_from_url(&mut data);
			if let Some(storage) = storage{
			  if let Some(data) = storage.get_string(DATA_KEY) {
				web_storage = serde_json::from_str(&data).unwrap();

			  }

			}

		match persistence::deserialize_from_url::<F32NumericTypes>() {
		  Ok(data)=>{
			entries_s = data;
		  },
		  Err(e)=>{
			serialization_error = Some(e);
		  }
		}
		match persistence::deserialize_from_url::<DefaultNumericTypes>() {
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
			persistence::load_file_entries(&cur_dir, &mut web_storage);
			  }
			}

		let mut next_color = 0;
		if entries_s.is_empty() {
			next_color = 1;
			entries_s.push(Entry::new_function(0, "sin(x)".to_string()));
		}
		let ctx_s = Arc::new(RwLock::new(evalexpr::HashMapContext::new()));

		if entries_d.is_empty() {
			next_color = 1;
			entries_d.push(Entry::new_function(0, "sin(x)".to_string()));
		}
		let ctx_d = Arc::new(RwLock::new(evalexpr::HashMapContext::new()));

		Self {
			#[cfg(not(target_arch = "wasm32"))]
			_puffin:                                     run_puffin_server(),
			state_f32:                                   State {
				entries:     entries_s,
				ctx:         ctx_s,
				name:        String::new(),
				// points_cache: PointsCache::default(),
				clear_cache: true,
			},
			state_f64:                                   State {
				entries:     entries_d,
				ctx:         ctx_d,
				name:        String::new(),
				// points_cache: PointsCache::default(),
				clear_cache: true,
			},
			ui:                                          UiState {
				// animating: Arc::new(AtomicBool::new(true)),
				conf,
				scheduled_url_update: false,
				last_url_update: 0.0,
				web_storage,
				next_color,
				plot_bounds: PlotBounds::from_min_max([-2.0, -2.0], [2.0, 2.0]),
				reset_graph: false,

				cur_dir,
				serialization_error,
				stack_overflow_guard: Arc::new(AtomicIsize::new(0)),
				eval_errors: AHashMap::default(),
				selected_plot_item: None,
				f32_epsilon: f32::EPSILON as f64,
				f64_epsilon: f64::EPSILON,
				permalink_string: String::new(),
				file_to_remove: None,
				lines: Vec::with_capacity(512),
				points: Vec::with_capacity(512),
				polygons: Vec::with_capacity(512),
			},
		}
	}
}
impl App for Application {
	fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
		if self.ui.conf.dark_mode {
			ctx.set_visuals(Visuals::dark());
		} else {
			ctx.set_visuals(Visuals::light());
		}
		ctx.set_pixels_per_point(self.ui.conf.ui_scale);

		let use_f32 = self.ui.conf.use_f32;
		if use_f32 {
			side_panel(&mut self.state_f32, &mut self.ui, ctx, frame);
			graph_panel(&mut self.state_f32, &mut self.ui, ctx);
		} else {
			side_panel(&mut self.state_f64, &mut self.ui, ctx, frame);
			graph_panel(&mut self.state_f64, &mut self.ui, ctx);
		}
		if use_f32 != self.ui.conf.use_f32 {
			if use_f32 {
				let mut output = Vec::with_capacity(1024);
				if persistence::serialize_to(&mut output, &self.state_f32.entries).is_ok() {
					self.state_f64.entries = persistence::deserialize_from(&output).unwrap();
					self.state_f64.clear_cache = true;
					self.state_f64.name = self.state_f32.name.clone();
				}
			} else {
				let mut output = Vec::with_capacity(1024);
				if persistence::serialize_to(&mut output, &self.state_f64.entries).is_ok() {
					self.state_f32.entries = persistence::deserialize_from(&output).unwrap();
					self.state_f32.clear_cache = true;
					self.state_f32.name = self.state_f64.name.clone();
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

	fn auto_save_interval(&self) -> core::time::Duration { core::time::Duration::from_secs(5) }

	fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
		egui::Color32::from_rgba_unmultiplied(12, 12, 12, 180).to_normalized_gamma_f32()
	}

	fn persist_egui_memory(&self) -> bool { true }

	fn raw_input_hook(&mut self, _ctx: &egui::Context, _raw_input: &mut egui::RawInput) {}
}
fn side_panel<T: EvalexprNumericTypes>(
	state: &mut State<T>, ui_state: &mut UiState, ctx: &egui::Context, frame: &mut eframe::Frame,
) {
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
						state.entries.push(Entry::new_function(ui_state.next_color, String::new()));
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
					}
				});

				if ui.button("Clear all").clicked() {
					state.entries.clear();
					state.clear_cache = true;
				}
			});

			ui.add_space(4.5);

			// let mut remove = None;
			let mut animating = false;
			for n in (0..state.entries.len()).rev() {
				let entry = &mut state.entries[n];
				let result = entry::edit_entry_ui(ui, entry, state.clear_cache);
				if result.remove {
					state.entries.remove(n);
				}
				animating |= result.animating;
				needs_recompilation |= result.needs_recompilation;
				if let Some(error) = result.error {
					ui_state.eval_errors.insert(n, error);
				}

				if let Some(eval_error) = ui_state.eval_errors.get(&n) {
					ui.label(RichText::new(eval_error).color(Color32::RED));
				}
				ui.separator();
			}

			// ui_state.animating.store(animating, Ordering::Relaxed);

			#[cfg(not(target_arch = "wasm32"))]
			ui.hyperlink_to("View Online", {
				let mut base_url = "https://vdrn.github.io/rust_graph/".to_string();
				base_url.push_str(ui_state.permalink_string.as_str());

				base_url
			});
			ui.separator();

			CollapsingHeader::new("Settings").default_open(true).show(ui, |ui| {
				// ui.horizontal_top(|ui| {
				// 	let mut total_bytes = 0;
				// 	for entry in state.points_cache.iter() {
				// 		total_bytes += 16 * entry.1.capacity()
				// 	}
				// 	ui.label(format!(
				// 		"Points Cache is at least: {}MB",
				// 		total_bytes as f64 / (1024.0 * 1024.0)
				// 	));
				// 	if ui.button("Clear points cache").clicked() {
				// 		state.clear_cache = true;
				// 	}
				// });

				ui.separator();
				ui.checkbox(&mut ui_state.conf.use_f32, "Use f32");

				ui.separator();
				ui.add(Slider::new(&mut ui_state.conf.resolution, 10..=1000).text("Point Resolution"));

				ui.separator();
				ui.add(Slider::new(&mut ui_state.conf.ui_scale, 1.0..=3.0).text("Ui Scale"));
			});

			ui.separator();
			#[cfg(target_arch = "wasm32")]
			const PERSISTANCE_TYPE: &str = "Local Storage";
			#[cfg(not(target_arch = "wasm32"))]
			const PERSISTANCE_TYPE: &str = "Persistence";
			CollapsingHeader::new(PERSISTANCE_TYPE).default_open(true).show(ui, |ui| {
				persistence::persistence_ui(state, ui_state, ui, frame);
			});

			if let Some(error) = &ui_state.serialization_error {
				if ui.label(RichText::new(error).color(Color32::RED)).clicked() {
					ui_state.serialization_error = None;
				}
			}

			ui.separator();
			ui.hyperlink_to("Github", "https://github.com/vdrn/rust_graph");

			if needs_recompilation || state.clear_cache {
				puffin::profile_scope!("clear_cache");
				// state.points_cache.clear();
				let mut output = Vec::new();
				if persistence::serialize_to(&mut output, &state.entries).is_ok() {
					// let mut data_str = String::with_capacity(output.len() + 1);
					let url_encoded = urlencoding::encode(str::from_utf8(&output).unwrap());
					let mut permalink_string = String::with_capacity(url_encoded.len() + 1);
					permalink_string.push('#');
					permalink_string.push_str(&url_encoded);
					ui_state.permalink_string = permalink_string;
					ui_state.scheduled_url_update = true;
				}

				state.ctx.write().unwrap().clear();
				state.ctx.write().unwrap().set_builtin_functions_disabled(false).unwrap();
				init_functions::<T>(&mut *(state.ctx.write().unwrap()));
				init_consts::<T>(&mut *(state.ctx.write().unwrap()));

				// state.context_stash.lock().unwrap().clear();

				for entry in state.entries.iter_mut() {
					entry::recompile_entry(entry, &state.ctx, &ui_state.stack_overflow_guard);
				}
			} else if animating {
				for entry in state.entries.iter_mut() {
					puffin::profile_scope!("entry_process");
					if entry.name != "x" {
						if let EntryType::Constant { value, .. } = &mut entry.ty {
							if !entry.name.is_empty() {
								state
									.ctx
									.write()
									.unwrap()
									.set_value(entry.name.as_str(), evalexpr::Value::<T>::Float(*value))
									.unwrap();
							}
						}
					}
				}
			}
			state.clear_cache = false;

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
		});
	});
	ui_state.eval_errors.clear();
}
fn graph_panel<T: EvalexprNumericTypes>(state: &mut State<T>, ui_state: &mut UiState, ctx: &egui::Context) {
	ui_state.lines.clear();
	ui_state.points.clear();
	ui_state.polygons.clear();

	let main_context = &*state.ctx.read().unwrap();
	let first_x = ui_state.plot_bounds.min()[0];
	let last_x = ui_state.plot_bounds.max()[0];
	// let (first_x, last_x) = snap_range_to_grid(first_x, last_x, 10.0);
	let plot_width = last_x - first_x;
	let mut points_to_draw = ui_state.conf.resolution.max(1);
	let mut step_size = plot_width / points_to_draw as f64;
	while points_to_draw > 2 && first_x + step_size == first_x {
		points_to_draw /= 2;
		step_size = plot_width / points_to_draw as f64;
	}
	let plot_params = entry::PlotParams {
		eps: if ui_state.conf.use_f32 { ui_state.f32_epsilon } else { ui_state.f64_epsilon },
		first_x,
		last_x,
		step_size,
	};

	// let animating = ui_state.animating.load(Ordering::Relaxed);
	puffin::profile_scope!("graph");

	for (ei, entry) in state.entries.iter_mut().enumerate() {
		ui_state.stack_overflow_guard.store(0, Ordering::Relaxed);
		let id = Id::new(ei);
		if let Err(error) = entry::create_entry_plot_elements(
			entry,
			id,
			Some(id) == ui_state.selected_plot_item,
			main_context,
			&plot_params,
			&mut ui_state.polygons,
			&mut ui_state.lines,
			&mut ui_state.points,
		) {
			ui_state.eval_errors.insert(ei, error);
		}
	}

	for (implicit, line_id, line) in ui_state.lines.iter() {
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
			for (other_implicit, other_line_id, other_line) in ui_state.lines.iter() {
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
				if let Some((t, y)) =
					implicit_intersects(point_before.y, point.y, other_point_before.y, other_point.y)
				{
					ui_state.points.push(
						Points::new("", [point_before.x + (point.x - point_before.x) * t, y])
							.color(Color32::GRAY)
							.radius(3.5),
					);
				}
			}
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
			puffin::profile_scope!("graph_lines");
			for poly in ui_state.polygons.drain(..) {
				plot_ui.polygon(poly);
			}
			for (_, _, line) in ui_state.lines.drain(..) {
				plot_ui.line(line);
			}
			for point in ui_state.points.drain(..) {
				plot_ui.points(point);
			}
		});

		if plot_res.response.double_clicked() {
			ui_state.reset_graph = true;
		}

		ui_state.plot_bounds = *plot_res.transform.bounds();
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

	if !(0.0..=1.0).contains(&x) {
		return None;
	}

	let y = p1_y + m1 * x;

	Some((x, y))
}

// pub fn snap_range_to_grid(start: f64, end: f64, expand_percent: f64) -> (f64, f64) {
// 	let range = end - start;
// 	let expansion = range * (expand_percent / 100.0);

// 	// Expand the range
// 	let expanded_start = start - expansion;
// 	let expanded_end = end + expansion;
// 	let expanded_range = expanded_end - expanded_start;

// 	let grid_size = calculate_grid_size_dynamic(expanded_range);

// 	// snap to boundaries
// 	let snapped_start = (expanded_start / grid_size).floor() * grid_size;
// 	let snapped_end = (expanded_end / grid_size).ceil() * grid_size;

// 	(snapped_start, snapped_end)
// }

// /// Calculate grid size dynamically with hysteresis
// fn calculate_grid_size_dynamic(range: f64) -> f64 {
// 	let range = range.abs();
// 	if range == 0.0 {
// 		return 1.0;
// 	}

// 	let exp = range.log10().floor();
// 	let power = 10_f64.powf(exp);
// 	let normalized = range / power;

// 	// Use wider buckets to reduce threshold sensitivity
// 	let multiplier = if normalized < 2.5 {
// 		0.5
// 	} else if normalized < 5.0 {
// 		1.0
// 	} else if normalized < 7.5 {
// 		2.0
// 	} else {
// 		5.0
// 	};
// 	multiplier * power
// }
