extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::cell::{Cell, RefCell};
use rustc_hash::FxHashMap;
use std::env;
use std::sync::Mutex;

use eframe::egui::{
	self, Align, Button, CollapsingHeader, Id, RichText, ScrollArea, SidePanel, Slider, TextEdit, TextStyle, Visuals, Window
};
use eframe::epaint::Color32;
use eframe::{App, CreationContext};
use egui_plot::{
	HLine, Legend, Plot, PlotBounds, PlotGeometry, PlotItem, PlotPoint, PlotTransform, Points, VLine
};
use evalexpr::{
	CompiledNode, Context, ContextWithMutableFunctions, ContextWithMutableVariables, DefaultNumericTypes, EvalexprError, EvalexprFloat, EvalexprNumericTypes, EvalexprResult, F32NumericTypes, Value
};
use rayon::iter::{
	IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator
};
use serde::{Deserialize, Serialize};
use thread_local::ThreadLocal;

mod entry;
mod marching_squares;
mod persistence;
use crate::entry::{
	ConstantType, DragPoint, DrawPoint, Entry, EntryType, OtherPointType, PointEntry, PointInteraction, PointInteractionType, deriv_step, f64_to_float, f64_to_value
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
const MAX_FUNCTION_NESTING: isize = 500;

struct ThreadLocalContext<T: EvalexprNumericTypes> {
	cc_x:                 Cell<T::Float>,
	cc_y:                 Cell<T::Float>,
	stack_overflow_guard: Cell<isize>,
}

impl<T: EvalexprNumericTypes> Default for ThreadLocalContext<T> {
	fn default() -> Self {
		Self {
			cc_x:                 Cell::new(T::Float::ZERO),
			cc_y:                 Cell::new(T::Float::ZERO),
			stack_overflow_guard: Default::default(),
		}
	}
}

struct State<T: EvalexprNumericTypes> {
	entries:              Vec<Entry<T>>,
	ctx:                  evalexpr::HashMapContext<T>,
	default_bounds:       Option<PlotBounds>,
	// context_stash: &'static Mutex<Vec<evalexpr::HashMapContext<T>>>,
	name:                 String,
	// points_cache: PointsCache,
	clear_cache:          bool,
	thread_local_context: Arc<ThreadLocal<ThreadLocalContext<T>>>,
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
pub fn expect_function_argument_amount<NumericTypes: EvalexprNumericTypes>(
	actual: usize, expected: usize,
) -> EvalexprResult<(), NumericTypes> {
	if actual == expected {
		Ok(())
	} else {
		Err(EvalexprError::wrong_function_argument_amount(actual, expected))
	}
}

fn init_functions<T: EvalexprNumericTypes>(ctx: &mut evalexpr::HashMapContext<T>) {
	macro_rules! add_function {
		($ident:ident) => {
			ctx.set_function(
				stringify!($ident).to_string(),
				evalexpr::Function::new(move |_, v| {
					expect_function_argument_amount(v.len(), 1)?;
					let v: T::Float = v[0].as_float()?;
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
					expect_function_argument_amount(v.len(), 2)?;
					let v1: T::Float = v[0].as_float()?;
					let v2: T::Float = v[1].as_float()?;

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

	macro_rules! add_function_3 {
		($ident:ident) => {
			ctx.set_function(
				stringify!($ident).to_string(),
				evalexpr::Function::new(move |_, v| {
					expect_function_argument_amount(v.len(), 3)?;
					let v1: T::Float = v[0].as_float()?;
					let v2: T::Float = v[1].as_float()?;
					let v3: T::Float = v[2].as_float()?;

					Ok(Value::Float(v1.$ident(&v2, &v3)))
				}),
			)
			.unwrap();
		};
	}
	add_function_3!(clamp);

	ctx.set_function(
		"normaldist".to_string(),
		evalexpr::Function::new(move |_, v| {
			let zero = T::Float::f64_to_float(0.0);
			let one = T::Float::f64_to_float(1.0);
			if v.is_empty() {
				return Err(EvalexprError::wrong_function_argument_amount_range(0, 1..=3));
			}

			let (x, mean, std_dev) = if let Ok(x) = v[0].as_float() {
				(x, zero, one)
			} else {
				let x = v[0].as_float()?;
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
		evalexpr::Function::new(|_, v: &[Value<T>]| {
			expect_function_argument_amount(v.len(), 2)?;
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
		evalexpr::Function::new(|_, v: &[Value<T>]| {
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
  ("d(func(x))", "Derivative of the expression with respect to x."),
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

struct UiState {
	conf:        AppConfig,
	next_id:     u64,
	plot_bounds: PlotBounds,
	// data_aspect: f32,
	reset_graph: bool,

	cur_dir:             String,
	serialization_error: Option<String>,
	serialized_states:   BTreeMap<String, String>,
	eval_errors:         FxHashMap<u64, String>,
	parsing_errors:      FxHashMap<u64, String>,

	selected_plot_line:   Option<(Id, bool)>,
	dragging_point_i:     Option<entry::PointInteraction>,
	plot_mouese_pos:      Option<(egui::Pos2, PlotTransform)>,
	showing_custom_label: bool,

	f32_epsilon:          f64,
	f64_epsilon:          f64,
	permalink_string:     String,
	scheduled_url_update: bool,
	last_url_update:      f64,
	file_to_remove:       Option<String>,

	draw_buffer:  Box<ThreadLocal<RefCell<entry::DrawBuffer>>>,
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
		let mut serialized_states = BTreeMap::default();

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
            serialized_states = serde_json::from_str(&data).unwrap();
          }
        }

        match persistence::deserialize_from_url::<F32NumericTypes>(&mut next_id) {
          Ok((data, bounds))=>{
            entries_s = data;
            default_bounds_s = bounds;

          },
          Err(e)=>{
            serialization_error = Some(e);
          }
        }
        match persistence::deserialize_from_url::<DefaultNumericTypes>(&mut next_id) {
          Ok((data,bounds))=>{
            entries_d = data;
            default_bounds_d = bounds;
          },
          Err(e)=>{
            serialization_error = Some(e);
          }
        }
      } else {
        let cur_dir = env::home_dir()
          .and_then(|d| d.join("rust_graphs").to_str().map(|s| s.to_string()))
          .unwrap_or_default();
        persistence::load_file_entries(&cur_dir, &mut serialized_states);
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
				entries:              entries_s,
				ctx:                  ctx_s,
				name:                 String::new(),
				default_bounds:       default_bounds_s,
				// points_cache: PointsCache::default(),
				clear_cache:          true,
				thread_local_context: Arc::new(ThreadLocal::new()),
			},
			state_f64: State {
				entries:              entries_d,
				default_bounds:       default_bounds_d,
				ctx:                  ctx_d,
				name:                 String::new(),
				// points_cache: PointsCache::default(),
				clear_cache:          true,
				thread_local_context: Arc::new(ThreadLocal::new()),
			},
			ui: UiState {
				#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
				full_frame_scope: None,
				// animating: Arc::new(AtomicBool::new(true)),
				conf,
				scheduled_url_update: false,
				last_url_update: 0.0,
				serialized_states,
				next_id,
				plot_bounds: PlotBounds::from_min_max([-2.0, -2.0], [2.0, 2.0]),
				reset_graph: false,

				cur_dir,
				serialization_error,
				eval_errors: FxHashMap::default(),
				parsing_errors: FxHashMap::default(),
				selected_plot_line: None,
				showing_custom_label: false,
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
					let (entries, default_bounds) =
						persistence::deserialize_from_json(&output, &mut self.ui.next_id).unwrap();
					self.state_f64.entries = entries;
					self.state_f64.default_bounds = default_bounds;
					self.state_f64.clear_cache = true;
					self.state_f64.name = self.state_f32.name.clone();
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
					let (entries, default_bounds) =
						persistence::deserialize_from_json(&output, &mut self.ui.next_id).unwrap();
					self.state_f32.entries = entries;
					self.state_f32.default_bounds = default_bounds;
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
		storage.set_string(DATA_KEY, serde_json::to_string(&self.ui.serialized_states).unwrap());
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

fn add_new_entry_btn<T: EvalexprNumericTypes>(
	ui: &mut egui::Ui, next_id: &mut u64, entries: &mut Vec<Entry<T>>, can_add_folder: bool,
) -> bool {
	let mut needs_recompilation = false;

	if entries.len() >= 1000 {
		return false;
	}
	ui.menu_button("➕ Add", |ui| {
		let new_function = Entry::new_function(*next_id, String::new());
		if ui.button(new_function.type_name()).clicked() {
			entries.push(new_function);
			*next_id += 1;
			needs_recompilation = true;
		}
		let new_constant = Entry::new_constant(*next_id);
		if ui.button(new_constant.type_name()).clicked() {
			entries.push(new_constant);
			*next_id += 1;
			needs_recompilation = true;
		}
		let new_points = Entry::new_points(*next_id);
		if ui.button(new_points.type_name()).clicked() {
			entries.push(new_points);
			*next_id += 1;
			needs_recompilation = true;
		}
		let new_integral = Entry::new_integral(*next_id);
		if ui.button(new_integral.type_name()).clicked() {
			entries.push(new_integral);
			*next_id += 1;
		}
		let new_label = Entry::new_label(*next_id);
		if ui.button(new_label.type_name()).clicked() {
			entries.push(new_label);
			*next_id += 1;
		}
		if can_add_folder {
			let new_folder = Entry::new_folder(*next_id);
			if ui.button(new_folder.type_name()).clicked() {
				entries.push(new_folder);
				*next_id += 1;
			}
		}
	});

	needs_recompilation
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
				if add_new_entry_btn(ui, &mut ui_state.next_id, &mut state.entries, true) {
					needs_recompilation = true;
				}

				if ui.button("❌ Clear all").clicked() {
					state.entries.clear();
					state.clear_cache = true;
				}
			});

			ui.add_space(4.5);

			let mut remove = None;
			let mut animating = false;
			// println!("state.entries: {:?}", state.entries.iter().map(|e|e.id).collect::<Vec<_>>());
			egui_dnd::dnd(ui, "entries_dnd").show_vec(&mut state.entries, |ui, entry, handle, _state| {
				if let EntryType::Folder { entries } = &mut entry.ty {
					ui.vertical(|ui| {
						ui.horizontal(|ui| {
							handle.ui(ui, |ui| {
								ui.label("||");
							});
							let folder_symbol = if entry.visible { "📂" } else { "📁" };
							if ui
								.add(
									Button::new(RichText::new(folder_symbol).strong().monospace())
										.corner_radius(10),
								)
								.clicked()
							{
								entry.visible = !entry.visible;
							}

							ui.add(TextEdit::singleline(&mut entry.name).hint_text("name")).changed();
							ui.with_layout(egui::Layout::right_to_left(Align::LEFT), |ui| {
								if entries.is_empty() {
									if ui.button("X").clicked() {
										remove = Some(entry.id);
									}
								}
								if add_new_entry_btn(ui, &mut ui_state.next_id, entries, false) {
									needs_recompilation = true;
								}
							});
						});
						if entry.visible {
							let mut remove_from_folder = None;

							egui_dnd::dnd(ui, entry.id).show_vec(entries, |ui, entry, handle, _state| {
								ui.horizontal(|ui| {
									handle.ui(ui, |ui| {
										ui.label("    |");
									});
									ui.horizontal(|ui| {
										let fe_result = entry::edit_entry_ui(ui, entry, state.clear_cache);
										if fe_result.remove {
											remove_from_folder = Some(entry.id);
										}
										animating |= fe_result.animating;
										needs_recompilation |= fe_result.needs_recompilation;
										if let Some(error) = fe_result.error {
											ui_state.parsing_errors.insert(entry.id, error);
										} else if fe_result.parsed {
											ui_state.parsing_errors.remove(&entry.id);
										}
									});
								});
								if let Some(parsing_error) = ui_state.parsing_errors.get(&entry.id) {
									ui.label(RichText::new(parsing_error).color(Color32::RED));
								} else if let Some(eval_error) = ui_state.eval_errors.get(&entry.id) {
									ui.label(RichText::new(eval_error).color(Color32::RED));
								}
							});
							if let Some(id) = remove_from_folder {
								if let Some(index) = entries.iter().position(|e| e.id == id) {
									entries.remove(index);
								}
							}
						}
					});
				} else {
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
				}
				if let Some(parsing_error) = ui_state.parsing_errors.get(&entry.id) {
					ui.label(RichText::new(parsing_error).color(Color32::RED));
				} else if let Some(eval_error) = ui_state.eval_errors.get(&entry.id) {
					ui.label(RichText::new(eval_error).color(Color32::RED));
				}
				ui.separator();
			});

			if let Some(id) = remove {
				if let Some(index) = state.entries.iter().position(|e| e.id == id) {
					state.entries.remove(index);
				}
			}

			#[cfg(not(target_arch = "wasm32"))]
			ui.hyperlink_to("View Online", {
				let mut base_url = "https://rust-graph.netlify.app/".to_string();
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
					if let Err((id, e)) =
						entry::recompile_entry(entry, &mut state.ctx, &state.thread_local_context)
					{
						ui_state.parsing_errors.insert(id, e);
					}
				}
			} else if animating {
				for entry in state.entries.iter_mut() {
					scope!("entry_process");
					match &mut entry.ty {
						EntryType::Constant { value, .. } => {
							if !entry.name.is_empty() {
								state
									.ctx
									.set_value(entry.name.as_str(), evalexpr::Value::<T>::Float(*value))
									.unwrap();
							}
						},
						EntryType::Folder { entries } => {
							for entry in entries {
								if let EntryType::Constant { value, .. } = &mut entry.ty {
									if !entry.name.is_empty() {
										state
											.ctx
											.set_value(
												entry.name.as_str(),
												evalexpr::Value::<T>::Float(*value),
											)
											.unwrap();
									}
								}
							}
						},
						_ => {},
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
	let first_x = ui_state.plot_bounds.min()[0];
	let last_x = ui_state.plot_bounds.max()[0];
	let first_y = ui_state.plot_bounds.min()[1];
	let last_y = ui_state.plot_bounds.max()[1];

	// let (first_x, last_x) = snap_range_to_grid(first_x, last_x, 10.0);
	let plot_width = last_x - first_x;
	let plot_height = last_y - first_y;

	let mut points_to_draw = ui_state.conf.resolution.max(1);
	let mut step_size = plot_width / points_to_draw as f64;
	while points_to_draw > 2 && first_x + step_size == first_x {
		points_to_draw /= 2;
		step_size = plot_width / points_to_draw as f64;
	}

	let mut points_to_draw_y = ui_state.conf.resolution.max(1);
	let mut step_size_y = plot_height / points_to_draw as f64;
	while points_to_draw_y > 2 && first_y + step_size_y == first_y {
		points_to_draw_y /= 2;
		step_size_y = plot_height / points_to_draw_y as f64;
	}
	let plot_params = entry::PlotParams {
		eps: if ui_state.conf.use_f32 { ui_state.f32_epsilon } else { ui_state.f64_epsilon },
		first_x,
		last_x,
		first_y,
		last_y,
		step_size,
		step_size_y,
		resolution: ui_state.conf.resolution,
	};
	state.ctx.set_value(entry::STEP_X_NAME, f64_to_value(deriv_step(plot_params.step_size))).unwrap();
	state.ctx.set_value(entry::STEP_Y_NAME, f64_to_value(deriv_step(plot_params.step_size_y))).unwrap();

	let main_context = &state.ctx;
	// let animating = ui_state.animating.load(Ordering::Relaxed);
	scope!("graph");

	let eval_errors = Mutex::new(&mut ui_state.eval_errors);

	// for stack in unsafe{&mut *(Arc::as_ptr(&ui_state.stack_overflow_guard) as *mut
	// ThreadLocal<Cell<isize>>)}.iter_mut(){   println!("stack :{}", stack.get());
	// }

	state.entries.par_iter_mut().enumerate().for_each(|(i, entry)| {
		scope!("entry_draw", entry.name.clone());

		let id = Id::new(entry.id);
		if let Err(errors) = entry::create_entry_plot_elements(
			entry,
			id,
			i as u32 * 1000,
			ui_state.selected_plot_line.map(|(id, _)| id),
			main_context,
			&plot_params,
			&ui_state.draw_buffer,
			&state.thread_local_context,
		) {
			for (id, error) in errors {
				eval_errors.lock().unwrap().insert(id, error);
			}
		}
	});

	let mut draw_lines = vec![];
	for draw_buffer in ui_state.draw_buffer.iter_mut() {
		let draw_buffer = draw_buffer.get_mut();
		draw_lines.append(&mut draw_buffer.lines);
	}
	let final_closest_point_to_mouse: Mutex<Option<((f64, f64), f64)>> = Mutex::new(None);
	let mouse_pos_in_graph = ui_state.plot_mouese_pos.map(|(pos, trans)| {
		let p = trans.value_from_position(pos);
		(p.x, p.y)
	});
	if let Some((selected_fline_id, show_closest_point_to_mouse)) = ui_state.selected_plot_line {
		draw_lines.par_iter().for_each(|fline| {
			let PlotGeometry::Points(plot_points) = fline.line.geometry() else {
				return;
			};
			let mut draw_buffer =
				ui_state.draw_buffer.get_or(|| RefCell::new(entry::DrawBuffer::new())).borrow_mut();
			if fline.id == selected_fline_id {
				let mut closest_point_to_mouse: Option<((f64, f64), f64)> = None;

				// Find local optima
				let mut prev: [Option<(f64, f64)>; 2] = [None; 2];
				for (pi, point) in plot_points.iter().enumerate() {
					let cur = (point.x, point.y);
					fn less_then(a: f64, b: f64, e: f64) -> bool { b - a > e }
					fn greater_then(a: f64, b: f64, e: f64) -> bool { a - b > e }
					if let (Some(prev_0), Some(prev_1)) = (prev[0], prev[1]) {
						// Find closest point to mouse
						if show_closest_point_to_mouse && let Some(mouse_pos_in_graph) = mouse_pos_in_graph {
							let mouse_on_seg = closest_point_on_segment(
								(prev_0.0, prev_0.1),
								(prev_1.0, prev_1.1),
								mouse_pos_in_graph,
							);
							let dist_sq = dist_sq(mouse_on_seg, mouse_pos_in_graph);
							if let Some(cur_closest_point_to_mouse) = closest_point_to_mouse {
								if dist_sq < cur_closest_point_to_mouse.1 {
									closest_point_to_mouse = Some((mouse_on_seg, dist_sq));
								}
							} else {
								closest_point_to_mouse = Some((mouse_on_seg, dist_sq));
							}
						}

						if prev_0.1.signum() != prev_1.1.signum() {
							// intersection with x axis
							let sum = prev_0.1.abs() + prev_1.1.abs();
							let x = if sum == 0.0 {
								(prev_0.0 + prev_1.0) * 0.5
							} else {
								let t = prev_0.1.abs() / sum;
								prev_0.0 + t * (prev_1.0 - prev_0.0)
							};
							draw_buffer.points.push(DrawPoint::new(
								fline.sorting_index,
								pi as u32,
								PointInteraction {
									x,
									y: 0.0,
									radius: fline.width,
									ty: PointInteractionType::Other(OtherPointType::IntersectionWithXAxis),
								},
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
							draw_buffer.points.push(DrawPoint::new(
								fline.sorting_index,
								pi as u32,
								PointInteraction {
									x: 0.0,
									y,
									radius: fline.width,
									ty: PointInteractionType::Other(OtherPointType::IntersectionWithYAxis),
								},
								Points::new("", [0.0, y]).color(Color32::GRAY).radius(fline.width),
							));
						}

						if less_then(prev_0.1, prev_1.1, plot_params.eps)
							&& greater_then(prev_1.1, cur.1, plot_params.eps)
						{
							// local maximum
							draw_buffer.points.push(DrawPoint::new(
								fline.sorting_index,
								pi as u32,
								PointInteraction {
									x:      prev_1.0,
									y:      prev_1.1,
									radius: fline.width,
									ty:     PointInteractionType::Other(OtherPointType::Maxima),
								},
								Points::new("", [prev_1.0, prev_1.1]).color(Color32::GRAY).radius(fline.width),
							));
						}
						if greater_then(prev_0.1, prev_1.1, plot_params.eps)
							&& less_then(prev_1.1, cur.1, plot_params.eps)
						{
							// local minimum
							draw_buffer.points.push(DrawPoint::new(
								fline.sorting_index,
								pi as u32,
								PointInteraction {
									x:      prev_1.0,
									y:      prev_1.1,
									radius: fline.width,
									ty:     PointInteractionType::Other(OtherPointType::Minima),
								},
								Points::new("", [prev_1.0, prev_1.1]).color(Color32::GRAY).radius(fline.width),
							));
						}
					}

					prev[0] = prev[1];
					prev[1] = Some(cur);
				}
				if show_closest_point_to_mouse {
					if let Some((closest_point, dist_sq)) = closest_point_to_mouse {
						let mut final_closest_point_to_mouse = final_closest_point_to_mouse.lock().unwrap();
						if let Some(cur_closest_point) = *final_closest_point_to_mouse {
							if dist_sq < cur_closest_point.1 {
								*final_closest_point_to_mouse = Some((closest_point, dist_sq));
							}
						} else {
							*final_closest_point_to_mouse = Some((closest_point, dist_sq));
						}
					}
				}
			} else {
				// find intersections
				let mut pi = 0;
				for selected_line in draw_lines.iter().filter(|sel| sel.id == selected_fline_id) {
					let PlotGeometry::Points(sel_points) = selected_line.line.geometry() else {
						continue;
					};
					for plot_seg in plot_points.windows(2) {
						for sel_seg in sel_points.windows(2) {
							if let Some(point) = intersect_segs(
								plot_seg[0], plot_seg[1], sel_seg[0], sel_seg[1], plot_params.eps,
							) {
								draw_buffer.points.push(DrawPoint::new(
									fline.sorting_index,
									pi as u32,
									PointInteraction {
										x:      point.x,
										y:      point.y,
										radius: fline.width,
										ty:     PointInteractionType::Other(OtherPointType::Intersection),
									},
									Points::new("", [point.x, point.y])
										.color(Color32::GRAY)
										.radius(selected_line.width),
								));
								pi += 1;
							}
						}
					}
				}
			}
		});
	}

	let mut draw_points = vec![];
	let mut draw_polygons = vec![];
	let mut draw_texts = vec![];
	for draw_buffer in ui_state.draw_buffer.iter_mut() {
		let draw_buffer = draw_buffer.get_mut();
		draw_polygons.append(&mut draw_buffer.polygons);
		draw_points.append(&mut draw_buffer.points);
		draw_texts.append(&mut draw_buffer.texts);
	}

	let closest_point_to_mouse = final_closest_point_to_mouse.into_inner().unwrap();
	if let Some(closest_point_to_mouse) = closest_point_to_mouse {
		draw_points.push(DrawPoint::new(
			0,
			0,
			PointInteraction {
				x:      closest_point_to_mouse.0.0,
				y:      closest_point_to_mouse.0.1,
				radius: 5.0,
				ty:     PointInteractionType::Other(OtherPointType::Point),
			},
			Points::new("", [closest_point_to_mouse.0.0, closest_point_to_mouse.0.1])
				.color(Color32::GRAY)
				.radius(5.0),
		));
	}

	draw_lines.sort_unstable_by_key(|draw_line| draw_line.sorting_index);
	draw_points.sort_unstable_by_key(|draw_point| draw_point.sorting_index);
	draw_polygons.sort_unstable_by_key(|draw_poly_group| draw_poly_group.sorting_index);
	draw_texts.sort_unstable_by_key(|draw_text| draw_text.sorting_index);

	egui::CentralPanel::default().show(ctx, |ui| {
		let plot_id = ui.make_persistent_id("Plot");
		let mut plot = Plot::new(plot_id)
			.id(plot_id)
			.legend(Legend::default().text_style(TextStyle::Body))
			.allow_drag(ui_state.dragging_point_i.is_none() && closest_point_to_mouse.is_none());

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
		if ui_state.showing_custom_label || closest_point_to_mouse.is_some() {
			// plot = plot.label_formatter(|_, _| String::new());
			plot = plot.show_x(false);
			plot = plot.show_y(false);

			ui_state.showing_custom_label = false;
		}

		// plot.center_y_axis
		let mut hovered_point: Option<(bool, PointInteraction)> = None;

		let plot_res = plot.show(ui, |plot_ui| {
			scope!("graph_show");
			plot_ui.hline(HLine::new("", 0.0).color(Color32::WHITE));
			plot_ui.vline(VLine::new("", 0.0).color(Color32::WHITE));
			for draw_poly_group in draw_polygons {
				for poly in draw_poly_group.polygons {
					plot_ui.polygon(poly);
				}
			}
			for draw_line in draw_lines {
				plot_ui.line(draw_line.line);
			}
			for draw_point in draw_points {
				if let Some((mouse_pos, transform)) = ui_state.plot_mouese_pos {
					let sel = &draw_point.interaction;
					let is_draggable = matches!(sel.ty, PointInteractionType::Draggable { .. });
					let sel_p = transform.position_from_point(&PlotPoint::new(sel.x, sel.y));
					let dist_sq = (sel_p.x - mouse_pos.x).powf(2.0) + (sel_p.y - mouse_pos.y).powf(2.0);

					if dist_sq < sel.radius * sel.radius {
						// ui_state.plot_custom_labels.push(((sel.x,sel.y), format!("Point {} {}",
						// sel.x,sel.y)));
						if let Some(current_hovered) = &hovered_point {
							if !current_hovered.0 {
								hovered_point = Some((is_draggable, sel.clone()));
							}
						} else {
							hovered_point = Some((is_draggable, sel.clone()));
						}
					}
				}
				plot_ui.points(draw_point.points);
			}
			for draw_text in draw_texts {
				plot_ui.text(draw_text.text);
			}
		});

		if plot_res.response.double_clicked() {
			ui_state.reset_graph = true;
		}

		ui_state.plot_bounds = *plot_res.transform.bounds();
		ui_state.plot_mouese_pos = plot_res.response.hover_pos().map(|p| {
			// let p = plot_res.transform.value_from_position(p);
			(p, plot_res.transform)
		});

		if let Some((is_draggable, hovered_point)) = &hovered_point {
			if *is_draggable && plot_res.response.drag_started() {
				ui_state.dragging_point_i = Some(hovered_point.clone());
			}
		}
		if plot_res.response.drag_stopped() {
			ui_state.dragging_point_i = None;
		}

		if let Some(dragging_point_i) = &ui_state.dragging_point_i {
			if let PointInteractionType::Draggable { i } = dragging_point_i.ty {
				if let Some((name, points)) = get_entry_mut_by_id(&mut state.entries, i.0).and_then(|entry| {
					let EntryType::Points { points, .. } = &mut entry.ty else {
						return None;
					};
					Some((&entry.name, points))
				}) {
					let point = &mut points[i.1 as usize];

					let point_x = dragging_point_i.x;
					let point_y = dragging_point_i.y;

					if let Some((x, y)) = point.val {
						let screen_x = plot_res.transform.position_from_point_x(x.to_f64());
						let screen_y = plot_res.transform.position_from_point_y(y.to_f64());

						ui_state.showing_custom_label = true;
						show_popup_label(
							ui,
							Id::new("drag_point_popup"),
							format!("Point {name}\nx: {x}\ny: {y}"),
							[screen_x, screen_y],
						);
					}

					if let (Some(drag_point_type), Some(screen_pos)) =
						(point.drag_point.clone(), plot_res.response.hover_pos())
					{
						let pos = plot_res.transform.value_from_position(screen_pos);

						match drag_point_type {
							DragPoint::BothCoordLiterals => {
								point.x.text = format!("{}", f64_to_float::<T>(pos.x));
								point.y.text = format!("{}", f64_to_float::<T>(pos.y));
							},
							DragPoint::XLiteral => {
								point.x.text = format!("{}", f64_to_float::<T>(pos.x));
							},
							DragPoint::YLiteral => {
								point.y.text = format!("{}", f64_to_float::<T>(pos.y));
							},
							DragPoint::XLiteralYConstant(y_const) => {
								point.x.text = format!("{}", f64_to_float::<T>(pos.x));
								let y_node = point.y.node.clone();
								drag(state, &y_const, y_node, point_y, pos.y, plot_params.eps);
							},
							DragPoint::YLiteralXConstant(x_const) => {
								point.y.text = format!("{}", f64_to_float::<T>(pos.y));
								let x_node = point.x.node.clone();
								drag(state, &x_const, x_node, point_x, pos.x, plot_params.eps);
							},
							DragPoint::BothCoordConstants(x_const, y_const) => {
								let x_node = point.x.node.clone();
								let y_node = point.y.node.clone();

								drag(state, &x_const, x_node, point_x, pos.x, plot_params.eps);
								drag(state, &y_const, y_node, point_y, pos.y, plot_params.eps);
							},
							DragPoint::XConstant(x_const) => {
								let x_node = point.x.node.clone();
								drag(state, &x_const, x_node, point_x, pos.x, plot_params.eps);
							},
							DragPoint::YConstant(y_const) => {
								let y_node = point.y.node.clone();
								drag(state, &y_const, y_node, point_y, pos.y, plot_params.eps);
							},
							DragPoint::SameConstantBothCoords(x_const) => {
								let x_node = point.x.node.clone();
								let y_node = point.y.node.clone();
								if let (Some(x_node), Some(y_node)) = (x_node, y_node) {
									if let Some(value) =
										find_constant_value(&mut state.entries, |e| e.name == x_const)
									{
										let new_value = solve_minimize(
											value.to_f64(),
											(pos.x, pos.y),
											plot_params.eps,
											|x| {
												state.ctx.set_value(&x_const, f64_to_value::<T>(x)).unwrap();
												(
													x_node
														.eval_float_with_context(&state.ctx)
														.unwrap()
														.to_f64(),
													y_node
														.eval_float_with_context(&state.ctx)
														.unwrap()
														.to_f64(),
												)
											},
										);
										*value = f64_to_float::<T>(new_value);
									}
								}
							},
						}
						state.clear_cache = true;
					}
				}
			}
		}

		if let Some((closest_point, _dist_sq)) = closest_point_to_mouse {
			let screen_x = plot_res.transform.position_from_point_x(closest_point.0);
			let screen_y = plot_res.transform.position_from_point_y(closest_point.1);
			ui_state.showing_custom_label = true;
			show_popup_label(
				ui,
				Id::new("point_on_fn"),
				format!("x:{}\ny: {}", f64_to_float::<T>(closest_point.0), f64_to_float::<T>(closest_point.1)),
				[screen_x, screen_y],
			);
		}

		if !ui_state.showing_custom_label
			&& let Some((_, hovered_point)) = &hovered_point
		{
			let screen_x = plot_res.transform.position_from_point_x(hovered_point.x);
			let screen_y = plot_res.transform.position_from_point_y(hovered_point.y);
			ui_state.showing_custom_label = true;
			show_popup_label(
				ui,
				Id::new("point_popup"),
				format!(
					"{}\nx:{}\ny: {}",
					hovered_point.name(),
					f64_to_float::<T>(hovered_point.x),
					f64_to_float::<T>(hovered_point.y)
				),
				[screen_x, screen_y],
			);
		}

		if let Some(hovered_id) = plot_res.hovered_plot_item {
			if plot_res.response.clicked()
				|| plot_res.response.drag_started()
				|| (ui_state.selected_plot_line.is_none() && plot_res.response.is_pointer_button_down_on())
			{
				// if ui.input(|i| i.pointer.primary_clicked()) {
				if state.entries.iter().any(|e| match &e.ty {
					EntryType::Function { .. } => Id::new(e.id) == hovered_id,
					EntryType::Folder { entries } => entries.iter().any(|e| Id::new(e.id) == hovered_id),
					_ => false,
				}) {
					// println!("selecting line {:?}", hovered_id);
					ui_state.selected_plot_line = Some((hovered_id, true));
				}
			}
		} else {
			if plot_res.response.clicked() || plot_res.response.drag_started() {
				if let Some(selected_plot_line) = &mut ui_state.selected_plot_line {
					if !selected_plot_line.1 {
						// println!("clearing selected line {:?}", ui_state.selected_plot_line);
						ui_state.selected_plot_line = None;
					}
				}
			}
		}

		if let Some(selected_plot_line) = &mut ui_state.selected_plot_line {
			if selected_plot_line.1 && !plot_res.response.is_pointer_button_down_on() {
				selected_plot_line.1 = false;
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
	state: &mut State<T>, name: &str, node: Option<CompiledNode<T>>, cur: f64, target: f64, eps: f64,
) -> bool {
	if let Some(point_node) = node {
		if let Some(value) = find_constant_value(&mut state.entries, |e| e.name == name) {
			if let Some(new_value) = solve_secant(value.to_f64(), cur, target, eps, |x| {
				state.ctx.set_value(name, f64_to_value::<T>(x)).unwrap();
				point_node.eval_float_with_context(&state.ctx).unwrap().to_f64()
			}) {
				// if let EntryType::Constant { value, .. } =
				// 	&mut get_entry_mut(&mut state.entries, c_idx).unwrap().ty
				// {
				*value = f64_to_float::<T>(new_value);
				// }
			} else {
				return false;
			}
		}
	}
	true
}

fn solve_secant(
	cur_x: f64, cur_y: f64, target_y: f64, eps: f64, mut f: impl FnMut(f64) -> f64,
) -> Option<f64> {
	// println!("secant cur_x: {cur_x} cur_y: {cur_y} target_y: {target_y}");
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
			// println!("nan or inf");
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
		// failed to converge
		// println!("failed to converge x0: {x0} x1: {x1} y0: {y0} y1: {y1}");
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

fn solve_minimize(
	cur_value: f64, mouse_pos: (f64, f64), eps: f64, mut f: impl FnMut(f64) -> (f64, f64),
) -> f64 {
	// Levenberg-Marquardt
	// minimizing abs(f(val) - mouse_pos)^2

	let mut t = cur_value;
	let max_iterations = 100;

	for _ in 0..max_iterations {
		let (fx, fy) = f(t);
		let residual_x = fx - mouse_pos.0;
		let residual_y = fy - mouse_pos.1;

		let error = residual_x * residual_x + residual_y * residual_y;

		if error < eps {
			// success
			return t;
		}

		let h = eps * 100.0;
		let (fx_h, fy_h) = f(t + h);

		// first derivative
		let df_dt_x = (fx_h - fx) / h;
		let df_dt_y = (fy_h - fy) / h;

		let gradient = 2.0 * (residual_x * df_dt_x + residual_y * df_dt_y);

		// second derivative (hessian)
		let (fx_2h, fy_2h) = f(t + 2.0 * h);
		let d2f_dt2_x = (fx_2h - 2.0 * fx_h + fx) / (h * h);
		let d2f_dt2_y = (fy_2h - 2.0 * fy_h + fy) / (h * h);
		// Hessian of dist sq
		let hessian =
			2.0 * (df_dt_x * df_dt_x + df_dt_y * df_dt_y + residual_x * d2f_dt2_x + residual_y * d2f_dt2_y);

		let damping = eps * 1000.0;
		let step = -gradient / (hessian.abs() + damping);

		// line search
		let mut best_t = t;

		for scale in [1.0, 0.5, 0.25, 0.1] {
			let t_new = t + scale * step;
			let (fx_new, fy_new) = f(t_new);
			let rx_new = fx_new - mouse_pos.0;
			let ry_new = fy_new - mouse_pos.1;
			let error_new = rx_new * rx_new + ry_new * ry_new;

			if error_new < error {
				best_t = t_new;
				break;
			}
		}

		if (best_t - t).abs() < eps {
			// we're stuck
			return t;
		}

		t = best_t;
	}

	t
}

fn show_popup_label(ui: &egui::Ui, id: Id, label: String, pos: [f32; 2]) {
	egui::Area::new(id).fixed_pos([pos[0] + 5.0, pos[1] + 5.0]).interactable(false).show(ui.ctx(), |ui| {
		egui::Frame::popup(ui.style()).show(ui, |ui| {
			ui.horizontal(|ui| ui.label(label));
			// ui.set_min_width(100.0);
		});
	});
}

fn find_constant_value<T: EvalexprNumericTypes>(
	entries: &mut [Entry<T>], pos_cb: impl Fn(&Entry<T>) -> bool,
) -> Option<&mut T::Float> {
	for entry in entries.iter_mut() {
		if pos_cb(entry) {
			if let EntryType::Constant { value, .. } = &mut entry.ty {
				return Some(value);
			}
		} else if let EntryType::Folder { entries } = &mut entry.ty {
			for sub_entry in entries.iter_mut() {
				if pos_cb(sub_entry) {
					if let EntryType::Constant { value, .. } = &mut sub_entry.ty {
						return Some(value);
					}
				}
			}
		}
	}
	None
}
fn get_entry_mut_by_id<T: EvalexprNumericTypes>(entries: &mut [Entry<T>], id: Id) -> Option<&mut Entry<T>> {
	for entry in entries.iter_mut() {
		if Id::new(entry.id) == id {
			return Some(entry);
		} else if let EntryType::Folder { entries } = &mut entry.ty {
			for sub_entry in entries.iter_mut() {
				if Id::new(sub_entry.id) == id {
					return Some(sub_entry);
				}
			}
		}
	}
	None
}
fn closest_point_on_segment(a: (f64, f64), b: (f64, f64), point: (f64, f64)) -> (f64, f64) {
	let (ax, ay) = a;
	let (bx, by) = b;
	let (px, py) = point;

	let ab_x = bx - ax;
	let ab_y = by - ay;

	let ap_x = px - ax;
	let ap_y = py - ay;

	let ab_len_sq = ab_x * ab_x + ab_y * ab_y;

	if ab_len_sq == 0.0 {
		// a == b
		return a;
	}

	let t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq;

	let t = t.clamp(0.0, 1.0);

	(ax + t * ab_x, ay + t * ab_y)
}
fn dist_sq(a: (f64, f64), b: (f64, f64)) -> f64 {
	let (ax, ay) = a;
	let (bx, by) = b;
	let a = ax - bx;
	let b = ay - by;
	a * a + b * b
}

#[cold]
fn cold() {}

pub fn unlikely(x: bool) -> bool {
	if x {
		cold()
	}
	x
}
