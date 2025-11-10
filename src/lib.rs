extern crate alloc;
use alloc::sync::Arc;
use core::cell::{Cell, RefCell};
use rustc_hash::FxHashMap;
use std::env;
use std::sync::Mutex;
use thread_local::ThreadLocal;

use eframe::egui::{
	self, CollapsingHeader, Id, RichText, ScrollArea, SidePanel, Slider, TextStyle, Visuals, Window
};
use eframe::epaint::Color32;
use eframe::{App, CreationContext};
use egui_plot::{
	HLine, Legend, Plot, PlotBounds, PlotGeometry, PlotItem, PlotPoint, PlotTransform, Points, VLine
};
use evalexpr::{
	Context, ContextWithMutableFunctions, ContextWithMutableVariables, DefaultNumericTypes, EvalexprError, EvalexprFloat, EvalexprNumericTypes, F32NumericTypes, Node, Value
};
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

mod entry;
mod marching_squares;
mod persistence;
use crate::entry::{
	ConstantType, DragPoint, Entry, EntryType, FunctionLine, PointEntry, f64_to_float, f64_to_value
};

#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
macro_rules! scope {
  ($($tt:tt)*) => {
      puffin::profile_scope!($($tt)*)
  }
}
#[cfg(not(all(feature = "puffin", not(target_arch = "wasm32"))))]
macro_rules! scope {
	($($tt:tt)*) => {};
}
pub(crate) use scope;
pub const DEFAULT_RESOLUTION: usize = 500;

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::{self, prelude::*};

#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

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
			.start(canvas, web_options, Box::new(|cc| Ok(Box::new(Application::new(cc)))))
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
	entries:        Vec<Entry<T>>,
	ctx:            evalexpr::HashMapContext<T>,
	default_bounds: Option<PlotBounds>,
	// context_stash: &'static Mutex<Vec<evalexpr::HashMapContext<T>>>,
	name:           String,
	// points_cache: PointsCache,
	clear_cache:    bool,
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
#[rustfmt::skip]
const BUILTIN_CONSTS: &[(&str, &str)] = &[
  ("e","2.718281828459045"),
  ("pi","3.141592653589793"),
  ("tau","6.283185307179586"),
];
fn init_functions<T: EvalexprNumericTypes>(ctx: &mut evalexpr::HashMapContext<T>) {
	macro_rules! add_function {
		($ident:ident) => {
			ctx.set_function(
				stringify!($ident).to_string(),
				evalexpr::Function::new(move |_, v| {
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

	add_function!(signum);
	add_function!(abs);

	macro_rules! add_function_2 {
		($ident:ident) => {
			ctx.set_function(
				stringify!($ident).to_string(),
				evalexpr::Function::new(move |_, v| {
					let v = v.as_tuple_ref()?;
					let v1: T::Float = v
						.first()
						.ok_or_else(|| {
							EvalexprError::CustomMessage("Expected 2 arguments, got 0.".to_string())
						})?
						.as_float()?;
					let v2: T::Float = v
						.get(1)
						.ok_or_else(|| {
							EvalexprError::CustomMessage("Expected 2 arguments, got 1.".to_string())
						})?
						.as_float()?;

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

	ctx.set_function(
		"normaldist".to_string(),
		evalexpr::Function::new(move |_, v| {
			let zero = T::Float::f64_to_float(0.0);
			let one = T::Float::f64_to_float(1.0);

			let (x, mean, std_dev) = if let Ok(x) = v.as_float() {
				(x, zero, one)
			} else {
				let v = v.as_tuple_ref()?;
				let x = v
					.first()
					.ok_or_else(|| {
						EvalexprError::CustomMessage("normaldist requires at least 1 argument".to_string())
					})?
					.as_float()?;
				let mean: T::Float =
					v.get(1).unwrap_or(&Value::Float(T::Float::f64_to_float(0.0))).as_float()?;
				let std_dev: T::Float =
					v.get(2).unwrap_or(&Value::Float(T::Float::f64_to_float(1.0))).as_float()?;
				(x, mean, std_dev)
			};

			let two = T::Float::f64_to_float(2.0);
			let coefficient = T::Float::f64_to_float(1.0)
				/ (std_dev * T::Float::f64_to_float(2.0 * core::f64::consts::PI).sqrt());
			let diff: T::Float = x - mean;
			let exponent = -(diff.pow(&two)) / (T::Float::f64_to_float(2.0) * std_dev.pow(&two));
			Ok(Value::Float(coefficient * exponent.exp()))
		}),
	)
	.unwrap();

	ctx.set_function(
		"g".to_string(),
		evalexpr::Function::new(|_, v: &Value<T>| {
			let v = v.as_fixed_len_tuple_ref(2)?;
			let tuple = v[0].as_tuple_ref()?;
			let index: T::Float = v[1].as_float()?;

			let index = index.to_f64() as usize;

			let value = tuple.get(index).ok_or_else(|| {
				EvalexprError::CustomMessage(format!(
					"Index out of bounds: index = {index} but the length was {}",
					tuple.len()
				))
			})?;
			Ok(value.clone())
		}),
	)
	.unwrap();
	ctx.set_function(
		"get".to_string(),
		evalexpr::Function::new(|_, v: &Value<T>| {
			let v = v.as_fixed_len_tuple_ref(2)?;
			let tuple = v[0].as_tuple_ref()?;
			let index: T::Float = v[1].as_float()?;

			let index = index.to_f64() as usize;

			let value = tuple.get(index).ok_or_else(|| {
				EvalexprError::CustomMessage(format!(
					"Index out of bounds: index = {index} but the length was {}",
					tuple.len()
				))
			})?;
			Ok(value.clone())
		}),
	)
	.unwrap();
	// };
}
#[rustfmt::skip]
const BUILTIN_FUNCTIONS: &[(&str, &str)] = &[
	("if(bool_expr,true_expr,false_expr)", " If the bool_expr is true, then evaluate the true_expr, otherwise evaluate the false_expr.",),
  ("get(tuple,index)", " Get the value at the index from the tuple."),
  ("g(tuple, index)", " Alias for `get`."),
	("", ""),
	("max(a, b)", " Returns the maximum of the two numbers."),
	("min(a, b)", " Returns the minimum of the two numbers."),
	("floor(a)", " Returns the largest integer less than or equal to a."),
	("round(a)", " Returns the nearest integer to a. If a value is half-way between two integers, round away from 0.0.",),
	("ceil(a)", "Returns the smallesst integer greater than or equal to a."),
	("singnum(a)", " Returns the sign of a."),
	("abs(a)", " Returns the absolute value of a."),
	("", ""),
	("ln(a)", " Compute the natural logarithm."),
	("ln2(a)", " Compute the logarithm base 2."),
	("ln10(a)", " Compute the logarithm base 10."),
	("log(a,base)", " Compute the logarithm to a certain base."),
	("exp(a)", " Exponentiate with base e."),
	("exp2(a)", " Exponentiate with base 2."),
	("pow(a,b)", " Compute the power of a to the exponent b."),
	("sqrt(a)", " Compute the square root."),
	("cbrt(a)", " Compute the cubic root."),
	("", ""),
	("cos(a)", " Compute the cosine."),
	("cosh(a)", " Compute the hyperbolic cosine."),
	("acos(a)", " Compute the arccosine."),
	("acosh(a)", " Compute the hyperbolic arccosine."),
	("sin(a)", " Compute the sine."),
	("sinh(a)", " Compute the hyperbolic sine."),
	("asin(a)", " Compute the arcsine."),
	("asinh(a)", " Compute the hyperbolic arcsine."),
	("tan(a)", " Compute the tangent."),
	("tanh(a)", " Compute the hyperbolic tangent."),
	("atan(a)", " Compute the arctangent."),
	("atanh(a)", " Compute the hyperbolic arctangent."),
	("atan2(a,b)", " Compute the four quadrant arctangent."),
	("", ""),
	("hypot(a,b)", " Compute the distance between the origin and a point (a,b) on the Euclidean plane."),

	("", ""),
  ("normaldist(a, mean, std_dev)", " Compute the probability density of normal distribution at a with mean(default 0) and standard deviation (default 1)."),

];
const OPERATORS: &[(&str, &str)] = &[
	("(...)", "Parentheses grouping"),
	("a + b", "Addition"),
	("a - b", "Subtraction"),
	("a * b", "Multiplication"),
	("a / b", "Division"),
	("a % b", "Modulo"),
	("a ^ b", "Power"),
	("-a", "Negation"),
	("f(param)", "Function call"),
	("f param", "Also a function call"),
];
const BOOLEAN_OPERATORS: &[(&str, &str)] = &[
	("a == b", "Equal to"),
	("a != b", "Not equal to"),
	("a < b", "Less than"),
	("a <= b", "Less than or equal to"),
	("a > b", "Greater than"),
	("a >= b", "Greater than or equal to"),
	("a && b", "Logical AND"),
	("a || b", "Logical OR"),
	("!a", "Logical NOT"),
];

#[derive(Serialize, Deserialize)]
struct AppConfig {
	dark_mode:  bool,
	use_f32:    bool,
	resolution: usize,
	fullscreen: bool,

	ui_scale: f32,
}
impl Default for AppConfig {
	fn default() -> Self {
		Self { fullscreen: true, dark_mode: true, use_f32: false, resolution: 500, ui_scale: 1.5 }
	}
}

struct SelectablePoint {
	i:      (Id, u32),
	x:      f64,
	y:      f64,
	radius: f32,
}
struct DrawBuffer {
	lines:    Vec<FunctionLine>,
	points:   Vec<(Option<SelectablePoint>, egui_plot::Points<'static>)>,
	polygons: Vec<egui_plot::Polygon<'static>>,
	texts:    Vec<egui_plot::Text>,
}
impl DrawBuffer {
	fn new() -> Self {
		Self {
			lines:    Vec::with_capacity(512),
			points:   Vec::with_capacity(512),
			polygons: Vec::with_capacity(512),
			texts:    Vec::with_capacity(8),
		}
	}
}
#[allow(clippy::non_send_fields_in_send_ty)]
/// SAFETY: Line/Points/Polygon are not Send/Sync because of `ExplicitGenerator` callbacks.
/// We dont use those so we're fine.
unsafe impl Send for DrawBuffer {}

struct UiState {
	conf:        AppConfig,
	next_id:     u64,
	plot_bounds: PlotBounds,
	// data_aspect: f32,
	reset_graph: bool,

	cur_dir:              String,
	serialization_error:  Option<String>,
	web_storage:          FxHashMap<String, String>,
	stack_overflow_guard: Arc<ThreadLocal<Cell<isize>>>,
	eval_errors:          FxHashMap<u64, String>,
	parsing_errors:       FxHashMap<u64, String>,

	selected_plot_line: Option<Id>,
	dragging_point:     Option<Id>,
	dragging_point_i:   Option<SelectablePoint>,
	plot_mouese_pos:    Option<((f32, f32), PlotTransform)>,

	f32_epsilon:          f64,
	f64_epsilon:          f64,
	permalink_string:     String,
	scheduled_url_update: bool,
	last_url_update:      f64,
	file_to_remove:       Option<String>,

	draw_buffer:  Box<ThreadLocal<RefCell<DrawBuffer>>>,
	showing_help: bool,

	#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
	full_frame_scope: Option<puffin::ProfilerScope>,
}
// #[derive(Clone, Debug)]
pub struct Application {
	state_f64: State<DefaultNumericTypes>,
	state_f32: State<F32NumericTypes>,
	ui:        UiState,
	#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
	_puffin:   puffin_http::Server,
}

#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
pub fn run_puffin_server() -> puffin_http::Server {
	let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
	let puffin_server = puffin_http::Server::new(&server_addr).unwrap();
	println!("Run this to view profiling data:  puffin_viewer {server_addr}");

	puffin::set_scopes_on(true);

	puffin_server
}
impl Application {
	#[allow(unused_mut)]
	pub fn new(cc: &CreationContext) -> Self {
		let mut entries_s = Vec::new();
		let mut entries_d = Vec::new();
		let mut default_bounds_s = None;
		let mut default_bounds_d = None;

		// TODO
		let mut web_storage = FxHashMap::default();

		let mut serialization_error = None;

		let conf: AppConfig = cc
			.storage
			.and_then(|s| s.get_string(CONF_KEY).and_then(|d| serde_json::from_str(&d).ok()))
			.unwrap_or_default();

		if !conf.fullscreen {
			cc.egui_ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
		}

		let mut next_id = 0;
		#[rustfmt::skip]
		cfg_if::cfg_if! {
      if #[cfg(target_arch = "wasm32")] {
        let cur_dir = String::new();
        // 	let error: Option<String> = web::get_data_from_url(&mut data);
        if let Some(storage) = cc.storage{
          if let Some(data) = storage.get_string(DATA_KEY) {
            web_storage = serde_json::from_str(&data).unwrap();
          }
        }

        match persistence::deserialize_from_url::<F32NumericTypes>() {
          Ok((data, bounds))=>{
            entries_s = data;
            default_bounds_s = bounds;
            next_id += entries_s.len() as u64;

          },
          Err(e)=>{
            serialization_error = Some(e);
          }
        }
        match persistence::deserialize_from_url::<DefaultNumericTypes>() {
          Ok((data,bounds))=>{
            entries_d = data;
            default_bounds_d = bounds;
            next_id += entries_s.len() as u64;
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

		if entries_s.is_empty() {
			next_id += 1;
			entries_s.push(Entry::new_function(0, "sin(x)".to_string()));
		}
		let ctx_s = evalexpr::HashMapContext::new();

		if entries_d.is_empty() {
			next_id += 1;
			entries_d.push(Entry::new_function(0, "sin(x)".to_string()));
		}
		let ctx_d = evalexpr::HashMapContext::new();

		Self {
			#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
			_puffin: run_puffin_server(),
			state_f32: State {
				entries:        entries_s,
				ctx:            ctx_s,
				name:           String::new(),
				default_bounds: default_bounds_s,
				// points_cache: PointsCache::default(),
				clear_cache:    true,
			},
			state_f64: State {
				entries:        entries_d,
				default_bounds: default_bounds_d,
				ctx:            ctx_d,
				name:           String::new(),
				// points_cache: PointsCache::default(),
				clear_cache:    true,
			},
			ui: UiState {
				#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
				full_frame_scope: None,
				// animating: Arc::new(AtomicBool::new(true)),
				conf,
				scheduled_url_update: false,
				last_url_update: 0.0,
				web_storage,
				next_id,
				plot_bounds: PlotBounds::from_min_max([-2.0, -2.0], [2.0, 2.0]),
				reset_graph: false,

				cur_dir,
				serialization_error,
				stack_overflow_guard: Arc::new(ThreadLocal::new()),
				eval_errors: FxHashMap::default(),
				parsing_errors: FxHashMap::default(),
				selected_plot_line: None,
				dragging_point: None,
				dragging_point_i: None,
				plot_mouese_pos: None,
				f32_epsilon: f32::EPSILON as f64,
				f64_epsilon: f64::EPSILON,
				permalink_string: String::new(),
				file_to_remove: None,
				draw_buffer: Box::new(ThreadLocal::new()),
				showing_help: false,
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
				if persistence::serialize_to_json(
					&mut output,
					&self.state_f32.entries,
					Some(&self.ui.plot_bounds),
				)
				.is_ok()
				{
					let (entries, default_bounds) = persistence::deserialize_from_json(&output).unwrap();
					self.state_f64.entries = entries;
					self.state_f64.default_bounds = default_bounds;
					self.state_f64.clear_cache = true;
					self.state_f64.name = self.state_f32.name.clone();
					self.ui.next_id += self.state_f32.entries.len() as u64;
				}
			} else {
				let mut output = Vec::with_capacity(1024);
				if persistence::serialize_to_json(
					&mut output,
					&self.state_f64.entries,
					Some(&self.ui.plot_bounds),
				)
				.is_ok()
				{
					let (entries, default_bounds) = persistence::deserialize_from_json(&output).unwrap();
					self.state_f32.entries = entries;
					self.state_f32.default_bounds = default_bounds;
					self.state_f32.clear_cache = true;
					self.state_f32.name = self.state_f64.name.clone();
					self.ui.next_id += self.state_f32.entries.len() as u64;
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
	#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
	{
		ui_state.full_frame_scope.take();
		puffin::GlobalProfiler::lock().new_frame();
		ui_state.full_frame_scope = puffin::profile_scope_custom!("full_frame");
	}

	scope!("side_panel");
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
				ui.menu_button("➕ Add", |ui| {
					let new_function = Entry::new_function(ui_state.next_id, String::new());
					if ui.button(new_function.type_name()).clicked() {
						state.entries.push(new_function);
						ui_state.next_id += 1;
						needs_recompilation = true;
					}
					let new_constant = Entry::new_constant(ui_state.next_id);
					if ui.button(new_constant.type_name()).clicked() {
						state.entries.push(new_constant);
						ui_state.next_id += 1;
						needs_recompilation = true;
					}
					let new_points = Entry::new_points(ui_state.next_id);
					if ui.button(new_points.type_name()).clicked() {
						state.entries.push(new_points);
						ui_state.next_id += 1;
						needs_recompilation = true;
					}
					let new_integral = Entry::new_integral(ui_state.next_id);
					if ui.button(new_integral.type_name()).clicked() {
						state.entries.push(new_integral);
						ui_state.next_id += 1;
					}
					let new_label = Entry::new_label(ui_state.next_id);
					if ui.button(new_label.type_name()).clicked() {
						state.entries.push(new_label);
						ui_state.next_id += 1;
					}
				});

				if ui.button("❌ Clear all").clicked() {
					state.entries.clear();
					state.clear_cache = true;
				}
			});

			ui.add_space(4.5);

			let mut remove = None;
			let mut animating = false;
			egui_dnd::dnd(ui, "entries_dnd").show_vec(&mut state.entries, |ui, entry, handle, _state| {
				ui.horizontal(|ui| {
					handle.ui(ui, |ui| {
						ui.label("||");
					});
					ui.horizontal(|ui| {
						let result = entry::edit_entry_ui(ui, entry, state.clear_cache);
						if result.remove {
							remove = Some(entry.id);
						}
						animating |= result.animating;
						needs_recompilation |= result.needs_recompilation;
						if let Some(error) = result.error {
							ui_state.parsing_errors.insert(entry.id, error);
						} else if result.parsed {
							ui_state.parsing_errors.remove(&entry.id);
						}
					});
				});
				if let Some(parsing_error) = ui_state.parsing_errors.get(&entry.id) {
					ui.label(RichText::new(parsing_error).color(Color32::RED));
				} else if let Some(eval_error) = ui_state.eval_errors.get(&entry.id) {
					ui.label(RichText::new(eval_error).color(Color32::RED));
				}
				ui.separator();
			});
			// for n in 0..state.entries.len() {
			// 	let entry = &mut state.entries[n];
			// }

			if let Some(id) = remove {
				if let Some(index) = state.entries.iter().position(|e| e.id == id) {
					state.entries.remove(index);
				}
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
				ui.add(Slider::new(&mut ui_state.conf.resolution, 10..=2000).text("Point Resolution"));

				ui.separator();
				ui.add(Slider::new(&mut ui_state.conf.ui_scale, 1.0..=3.0).text("Ui Scale"));

				ui.separator();
				#[cfg(not(target_arch = "wasm32"))]
				if ui.button("Toggle Fullscreen").clicked()
					|| ui.input(|i| i.key_pressed(egui::Key::F11))
					|| ui.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::Enter))
				{
					ui_state.conf.fullscreen = !ui_state.conf.fullscreen;
					ui.ctx().send_viewport_cmd(egui::ViewportCommand::Fullscreen(ui_state.conf.fullscreen));
				}

				if ui.button(format!("{} Help", if ui_state.showing_help { "📖" } else { "📚" })).clicked()
				{
					ui_state.showing_help = !ui_state.showing_help;
				}
				if ui_state.showing_help {
					Window::new("📖 Help").open(&mut ui_state.showing_help).show(ctx, |ui| {
						ScrollArea::vertical().show(ui, |ui| {
							ui.columns_const::<3, _>(|columns| {
								columns[0].heading("Operators");
								for &(name, value) in OPERATORS {
									columns[0].horizontal_wrapped(|ui| {
										ui.label(RichText::new(name).monospace().strong());
										ui.label(value);
									});
								}
								columns[1].heading("Boolean Operators");
								for &(name, value) in BOOLEAN_OPERATORS {
									columns[1].horizontal_wrapped(|ui| {
										ui.label(RichText::new(name).monospace().strong());
										ui.label(value);
									});
								}
								columns[2].heading("Builtin Constants");
								for &(name, value) in BUILTIN_CONSTS {
									columns[2].horizontal_wrapped(|ui| {
										ui.label(RichText::new(name).monospace().strong());
										ui.label(value);
									});
								}
							});

							ui.separator();
							ui.heading("Builtin Functions");
							for &(name, value) in BUILTIN_FUNCTIONS {
								if name.is_empty() {
									ui.separator();
								} else {
									ui.horizontal_wrapped(|ui| {
										ui.label(RichText::new(name).monospace().strong());
										ui.label(value);
									});
								}
							}
						});
					});
				}
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
				scope!("clear_cache");
				// state.points_cache.clear();
				match persistence::serialize_to_url(&state.entries, Some(&ui_state.plot_bounds)) {
					Ok(output) => {
						// let mut data_str = String::with_capacity(output.len() + 1);
						// let url_encoded = urlencoding::encode(str::from_utf8(&output).unwrap());
						let mut permalink_string = String::with_capacity(output.len() + 1);
						permalink_string.push('#');
						permalink_string.push_str(&output);
						ui_state.permalink_string = permalink_string;
						ui_state.scheduled_url_update = true;
					},
					Err(e) => {
						println!("Error: {e}");
					},
				}

				state.ctx.clear();
				state.ctx.set_builtin_functions_disabled(false).unwrap();
				init_functions::<T>(&mut state.ctx);
				init_consts::<T>(&mut state.ctx);

				// state.context_stash.lock().unwrap().clear();

				for entry in state.entries.iter_mut() {
					scope!("entry_recompile");
					if let Err(e) =
						entry::recompile_entry(entry, &mut state.ctx, &ui_state.stack_overflow_guard)
					{
						ui_state.parsing_errors.insert(entry.id, e);
					}
				}
			} else if animating {
				for entry in state.entries.iter_mut() {
					scope!("entry_process");
					if entry.name != "x" && entry.name != "y" {
						if let EntryType::Constant { value, .. } = &mut entry.ty {
							if !entry.name.is_empty() {
								state
									.ctx
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
	// ui_state.lines.clear();
	// ui_state.points.clear();
	// ui_state.polygons.clear();
	// ui_state.texts.clear();

	let main_context = &state.ctx;
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
		first_y: ui_state.plot_bounds.min()[1],
		last_y: ui_state.plot_bounds.max()[1],
		step_size,
		resolution: ui_state.conf.resolution,
	};

	// let animating = ui_state.animating.load(Ordering::Relaxed);
	scope!("graph");

	let eval_errors = Mutex::new(&mut ui_state.eval_errors);
	state.entries.par_iter_mut().for_each(|entry| {
		// for entry in state.entries.iter_mut() {
		scope!("entry_draw", entry.name.clone());
		ui_state.stack_overflow_guard.get_or_default().set(0);
		let draw_buffer = ui_state.draw_buffer.get_or(|| RefCell::new(DrawBuffer::new()));
		let mut draw_buffer = draw_buffer.borrow_mut();

		let id = Id::new(entry.id); //.with("entry_plot_els");
		if let Err(error) = entry::create_entry_plot_elements(
			entry, id, ui_state.selected_plot_line, main_context, &plot_params, &mut draw_buffer,
		) {
			eval_errors.lock().unwrap().insert(entry.id, error);
		}
	});

	let mut flines = vec![];
	for draw_buffer in ui_state.draw_buffer.iter_mut() {
		let draw_buffer = draw_buffer.get_mut();
		flines.reserve(draw_buffer.lines.len());
		for fline in draw_buffer.lines.drain(..) {
			// if Some(fline.id) == ui_state.selected_plot_item {
			// 	selected_fline = Some(i);
			// }
			flines.push(fline);
		}
	}
	if let Some(selected_fline_id) = ui_state.selected_plot_line {
		flines.par_iter().for_each(|fline| {
			let PlotGeometry::Points(plot_points) = fline.line.geometry() else {
				return;
			};
			let mut draw_buffer = ui_state.draw_buffer.get_or(|| RefCell::new(DrawBuffer::new())).borrow_mut();
			if fline.id == selected_fline_id {
				// Find local optima
				let mut prev: [Option<(f64, f64)>; 2] = [None; 2];
				for point in plot_points.iter() {
					let cur = (point.x, point.y);
					fn less_then(a: f64, b: f64, e: f64) -> bool { b - a > e }
					fn greater_then(a: f64, b: f64, e: f64) -> bool { a - b > e }
					if let (Some(prev_0), Some(prev_1)) = (prev[0], prev[1]) {
						if prev_0.1.signum() != prev_1.1.signum() {
							// intersection with x axis
							let sum = prev_0.1.abs() + prev_1.1.abs();
							let x = if sum == 0.0 {
								(prev_0.0 + prev_1.0) * 0.5
							} else {
								let t = prev_0.1.abs() / sum;
								prev_0.0 + t * (prev_1.0 - prev_0.0)
							};
							draw_buffer.points.push((
								None,
								Points::new("", [x, 0.0]).color(Color32::GRAY).radius(fline.width),
							));
						}

						if prev_0.0.signum() != prev_1.0.signum() {
							// intersection with y axis
							let sum = prev_0.0.abs() + prev_1.0.abs();
							let y = if sum == 0.0 {
								(prev_0.1 + prev_1.1) * 0.5
							} else {
								let t = prev_0.0.abs() / sum;
								prev_0.1 + t * (prev_1.1 - prev_0.1)
							};
							draw_buffer.points.push((
								None,
								Points::new("", [0.0, y]).color(Color32::GRAY).radius(fline.width),
							));
						}

						if less_then(prev_0.1, prev_1.1, plot_params.eps)
							&& greater_then(prev_1.1, cur.1, plot_params.eps)
						{
							// local maximum
							draw_buffer.points.push((
								None,
								Points::new("", [prev_1.0, prev_1.1]).color(Color32::GRAY).radius(fline.width),
							));
						}
						if greater_then(prev_0.1, prev_1.1, plot_params.eps)
							&& less_then(prev_1.1, cur.1, plot_params.eps)
						{
							// local minimum
							draw_buffer.points.push((
								None,
								Points::new("", [prev_1.0, prev_1.1]).color(Color32::GRAY).radius(fline.width),
							));
						}
					}

					prev[0] = prev[1];
					prev[1] = Some(cur);
				}
			} else {
				for selected_line in flines.iter().filter(|sel| sel.id == selected_fline_id) {
					let PlotGeometry::Points(sel_points) = selected_line.line.geometry() else {
						continue;
					};
					// println!("finding intersections between curent {:?} and selected {:?}", fline.id,
					// selected_line.id);
					for plot_seg in plot_points.windows(2) {
						for sel_seg in sel_points.windows(2) {
							if let Some(point) = intersect_segs(
								plot_seg[0], plot_seg[1], sel_seg[0], sel_seg[1], plot_params.eps,
							) {
								// println!("Found intersection {point:?}");
								draw_buffer.points.push((
									None,
									Points::new("", [point.x, point.y])
										.color(Color32::GRAY)
										.radius(selected_line.width),
								));
							}
						}
					}
				}
				// find intersections
			}
		});
	}

	egui::CentralPanel::default().show(ctx, |ui| {
		let plot_id = ui.make_persistent_id("Plot");
		let mut plot = Plot::new(plot_id)
			.id(plot_id)
			.legend(Legend::default().text_style(TextStyle::Body))
			.allow_drag(ui_state.dragging_point_i.is_none());

		let mut bounds = ui_state.plot_bounds;
		if ui_state.reset_graph {
			ui_state.reset_graph = false;
			plot = plot.data_aspect(1.0).center_x_axis(true).center_y_axis(true).reset();
			if let Some(default_bounds) = state.default_bounds {
				bounds = default_bounds;
			}
		}
		plot = plot
			.default_x_bounds(bounds.min()[0], bounds.max()[0])
			.default_y_bounds(bounds.min()[1], bounds.max()[1]);

		let mut hovered_point = None;
		let plot_res = plot.show(ui, |plot_ui| {
			scope!("graph_show");
			plot_ui.hline(HLine::new("", 0.0).color(Color32::WHITE));
			plot_ui.vline(VLine::new("", 0.0).color(Color32::WHITE));
			for draw_buffer in ui_state.draw_buffer.iter_mut() {
				let mut draw_buffer = draw_buffer.borrow_mut();
				for poly in draw_buffer.polygons.drain(..) {
					plot_ui.polygon(poly);
				}
			}
			for fline in flines.drain(..) {
				plot_ui.line(fline.line);
			}
			for draw_buffer in ui_state.draw_buffer.iter_mut() {
				let mut draw_buffer = draw_buffer.borrow_mut();
				for (sel, point) in draw_buffer.points.drain(..) {
					if let (Some(sel), Some((mouse_pos, transform))) = (sel, ui_state.plot_mouese_pos) {
						let sel_p = transform.position_from_point(&PlotPoint::new(sel.x, sel.y));
						let dist_sq = (sel_p.x - mouse_pos.0).powf(2.0) + (sel_p.y - mouse_pos.1).powf(2.0);
						if dist_sq < sel.radius * sel.radius {
							hovered_point = Some(sel);
						}
					}
					plot_ui.points(point);
				}
			}
			for draw_buffer in ui_state.draw_buffer.iter_mut() {
				let mut draw_buffer = draw_buffer.borrow_mut();
				for text in draw_buffer.texts.drain(..) {
					plot_ui.text(text);
				}
			}
		});

		if plot_res.response.double_clicked() {
			ui_state.reset_graph = true;
		}

		ui_state.plot_bounds = *plot_res.transform.bounds();
		ui_state.plot_mouese_pos = plot_res.response.hover_pos().map(|p| {
			// let p = plot_res.transform.value_from_position(p);
			((p.x, p.y), plot_res.transform)
		});

		if let Some(hovered_point) = hovered_point {
			// println!("hovered {:?}", hovered_point.i);
			if plot_res.response.drag_started() {
				// println!("drag started {:?}", hovered_point.i);
				ui_state.dragging_point_i = Some(hovered_point);
			}
		}
		if plot_res.response.drag_stopped() {
			// if let Some(dragging_point_i) = &ui_state.dragging_point_i {
			// 	println!("drag stopped {:?}", dragging_point_i.i);
			// }
			ui_state.dragging_point_i = None;
		}

		if let Some(dragging_point_i) = &ui_state.dragging_point_i {
			if let Some(points) = state
				.entries
				.iter_mut()
				.find(|e| Id::new(e.id) == dragging_point_i.i.0)
				.and_then(|e| if let EntryType::Points(points) = &mut e.ty { Some(points) } else { None })
			{
				let point = &mut points[dragging_point_i.i.1 as usize];

				let point_x = dragging_point_i.x;
				let point_y = dragging_point_i.y;

				if let (Some(drag_point_type), Some(screen_pos)) =
					(point.drag_point.clone(), plot_res.response.hover_pos())
				{
					let pos = plot_res.transform.value_from_position(screen_pos);

					match drag_point_type {
						DragPoint::BothCoordLiterals => {
							point.x.text = format!("{}", pos.x);
							point.y.text = format!("{}", pos.y);
						},
						DragPoint::XLiteral => {
							point.x.text = format!("{}", pos.x);
						},
						DragPoint::YLiteral => {
							point.y.text = format!("{}", pos.y);
						},
						DragPoint::XLiteralYConstant(y_const) => {
							point.x.text = format!("{}", pos.x);
							let x_node = point.x.node.clone();
							drag(state, y_const, x_node, point_y, pos.y, plot_params.eps);
						},
						DragPoint::YLiteralXConstant(x_const) => {
							point.y.text = format!("{}", pos.y);
							let x_node = point.x.node.clone();
							drag(state, x_const, x_node, point_x, pos.x, plot_params.eps);
						},
						DragPoint::BothCoordConstants(x_const, y_const) => {
							let x_node = point.x.node.clone();
							let y_node = point.y.node.clone();

							drag(state, x_const, x_node, point_x, pos.x, plot_params.eps);
							drag(state, y_const, y_node, point_y, pos.y, plot_params.eps);
						},
						DragPoint::XConstant(x_const) => {
							let x_node = point.x.node.clone();
							drag(state, x_const, x_node, point_x, pos.x, plot_params.eps);
						},
						DragPoint::YConstant(y_const) => {
							let y_node = point.y.node.clone();
							drag(state, y_const, y_node, point_y, pos.y, plot_params.eps);
						},
					}
					state.clear_cache = true;
				}
			}
		}

		if let Some(hovered_id) = plot_res.hovered_plot_item {
			if ui.input(|i| i.pointer.primary_clicked()) {
				if state.entries.iter().any(|e| {
					if matches!(e.ty, EntryType::Function { .. }) {
						Id::new(e.id) == hovered_id
					} else {
						false
					}
				}) {
					ui_state.selected_plot_line = Some(hovered_id);
				}
			}
		} else if ui_state.dragging_point.is_none() {
			if ui.input(|i| i.pointer.primary_clicked()) {
				ui_state.selected_plot_line = None;
			}
		}
	});
}

// fn implicit_intersects(p1_y: f64, p2_y: f64, o_p1_y: f64, o_p2_y: f64) -> Option<(f64, f64)> {
// 	let m1 = p2_y - p1_y;
// 	let m2 = o_p2_y - o_p1_y;

// 	// Check if lines are parallel
// 	let slope_diff = m1 - m2;
// 	if slope_diff.abs() < 1e-10 {
// 		return None;
// 	}

// 	// Line equations:
// 	// y1 = p1_y + m1 * x
// 	// y2 = o_p1_y + m2 * x

// 	let x = (o_p1_y - p1_y) / slope_diff;

// 	if !(0.0..=1.0).contains(&x) {
// 		return None;
// 	}

// 	let y = p1_y + m1 * x;

// 	Some((x, y))
// }

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
//
fn drag<T: EvalexprNumericTypes>(
	state: &mut State<T>, name: String, node: Option<Node<T>>, cur: f64, target: f64, eps: f64,
) {
	if let Some(point_node) = node {
		if let Some(c_idx) = state.entries.iter().position(|e| e.name == name) {
			if let EntryType::Constant { value, .. } = &state.entries[c_idx].ty {
				let value = *value;
				if let Some(new_value) = solve_secant(value.to_f64(), cur, target, eps, |x| {
					state.ctx.set_value(&name, f64_to_value::<T>(x)).unwrap();
					point_node.eval_float_with_context(&state.ctx).unwrap().to_f64()
				}) {
					if let EntryType::Constant { value, .. } = &mut state.entries[c_idx].ty {
						*value = f64_to_float::<T>(new_value);
					}
				}
			}
		}
	}
}

fn solve_secant(
	cur_x: f64, cur_y: f64, target_y: f64, eps: f64, mut f: impl FnMut(f64) -> f64,
) -> Option<f64> {
	// find root f(x) - target_y = 0
	let max_iterations = 40;
	let h = 0.0001; // small step for second point

	// initial points
	let mut x0 = cur_x;
	let mut y0 = cur_y - target_y;

	let mut x1 = cur_x + h;
	let mut y1 = f(x1) - target_y;

	for _ in 0..max_iterations {
		if y1.abs() < eps {
			// success
			return Some(x1);
		}

		if (y1 - y0) == 0.0 {
			// div by zero
			// We do best effort guess here.
			// TODO: this is not the solution!
			// The real solution is to disallow dragging of non-injective expressions.
			if (x0 - cur_x).abs() < (x1 - cur_x).abs() {
				// x0 is closer to cur_x
				return Some(x0);
			}
			// x1 is closer to cur_x
			return Some(x1);
		}

		let x_next = x1 - y1 * (x1 - x0) / (y1 - y0);
		if !x_next.is_finite() {
			// nan or inf
			return None;
		}

		x0 = x1;
		y0 = y1;
		x1 = x_next;
		y1 = f(x1) - target_y;
	}

	if y1.abs() < eps {
		Some(x1)
	} else {
		// Failed to converge
		None
	}
}
fn intersect_segs(a1: PlotPoint, a2: PlotPoint, b1: PlotPoint, b2: PlotPoint, eps: f64) -> Option<PlotPoint> {
	let d1x = a2.x - a1.x;
	let d1y = a2.y - a1.y;
	let d2x = b2.x - b1.x;
	let d2y = b2.y - b1.y;

	let denom = d1x * d2y - d1y * d2x;

	if denom.abs() < eps {
    // parallel
		return None;
	}

	let dx = b1.x - a1.x;
	let dy = b1.y - a1.y;

	let t = (dx * d2y - dy * d2x) / denom;
	let u = (dx * d1y - dy * d1x) / denom;

	if t >= -eps && t <= 1.0 + eps && u >= -eps && u <= 1.0 + eps {
		Some(PlotPoint { x: a1.x + t * d1x, y: a1.y + t * d1y })
	} else {
		None
	}
}
