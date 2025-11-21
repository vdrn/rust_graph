#![feature(adt_const_params)]
extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::cell::RefCell;
use std::env;

use eframe::egui::{self, Id, Visuals};
use eframe::{App, CreationContext};
use egui_plot::{PlotBounds, PlotTransform};
use evalexpr::{
	DefaultNumericTypes, EvalexprError, EvalexprFloat, EvalexprResult, F32NumericTypes, Stack, Value, istr
};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use thread_local::ThreadLocal;

mod app_ui;
mod draw_buffer;
mod entry;
mod marching_squares;
mod math;
mod persistence;

use entry::{ConstantType, Entry, EntryType, PointEntry};
use marching_squares::MarchingSquaresCache;

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

use crate::draw_buffer::DrawBufferRC;

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

#[repr(align(128))]
struct ThreadLocalContext<T: EvalexprFloat> {
	stack:                  RefCell<Stack<T>>,
	marching_squares_cache: MarchingSquaresCache,
}

impl<T: EvalexprFloat> Default for ThreadLocalContext<T> {
	fn default() -> Self {
		Self {
			stack:                  RefCell::new(Stack::<T>::with_capacity(128)),
			marching_squares_cache: MarchingSquaresCache::default(),
		}
	}
}

struct State<T: EvalexprFloat> {
	entries:              Vec<Entry<T>>,
	ctx:                  evalexpr::HashMapContext<T>,
	default_bounds:       Option<PlotBounds>,
	// context_stash: &'static Mutex<Vec<evalexpr::HashMapContext<T>>>,
	name:                 String,
	// points_cache: PointsCache,
	clear_cache:          bool,
	thread_local_context: Arc<ThreadLocal<ThreadLocalContext<T>>>,
}
fn init_consts<T: EvalexprFloat>(ctx: &mut evalexpr::HashMapContext<T>) {
	macro_rules! add_const {
		($ident:ident, $uppercase:expr, $lowercase:expr) => {
			ctx.set_value(istr($lowercase), Value::Float(T::f64_to_float(core::f64::consts::$ident))).unwrap();
			ctx.set_value(istr($uppercase), Value::Float(T::f64_to_float(core::f64::consts::$ident))).unwrap();
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
pub fn expect_function_argument_amount<NumericTypes: EvalexprFloat>(
	actual: usize, expected: usize,
) -> EvalexprResult<(), NumericTypes> {
	if actual == expected {
		Ok(())
	} else {
		Err(EvalexprError::wrong_function_argument_amount(actual, expected))
	}
}

fn init_functions<T: EvalexprFloat>(ctx: &mut evalexpr::HashMapContext<T>) {
	ctx.set_function(
		istr("normaldist"),
		evalexpr::RustFunction::new(move |s, _| {
			let zero = T::f64_to_float(0.0);
			let one = T::f64_to_float(1.0);
			if s.num_args() == 0 {
				return Err(EvalexprError::wrong_function_argument_amount_range(0, 1..=3));
			}

			let (x, mean, std_dev) = if let Ok(x) = s.get_arg(0).unwrap().as_float() {
				(x, zero, one)
			} else {
				let x = s.get_arg(0).unwrap().as_float()?;
				let mean: T = s.get_arg(1).unwrap_or(&Value::Float(T::f64_to_float(0.0))).as_float()?;
				let std_dev: T = s.get_arg(2).unwrap_or(&Value::Float(T::f64_to_float(1.0))).as_float()?;
				(x, mean, std_dev)
			};

			let two = T::f64_to_float(2.0);
			let coefficient =
				T::f64_to_float(1.0) / (std_dev * T::f64_to_float(2.0 * core::f64::consts::PI).sqrt());
			let diff: T = x - mean;
			let exponent = -(diff.pow(&two)) / (T::f64_to_float(2.0) * std_dev.pow(&two));
			Ok(Value::Float(coefficient * exponent.exp()))
		}),
	);
	ctx.set_function(
		istr("g"),
		evalexpr::RustFunction::new(|s, _| {
			expect_function_argument_amount(s.num_args(), 2)?;
			let tuple = s.get_arg(0).unwrap().as_tuple_ref()?;
			let index: T = s.get_arg(1).unwrap().as_float()?;

			let index = index.to_f64() as usize;

			let value = tuple.get(index).ok_or_else(|| {
				EvalexprError::CustomMessage(format!(
					"Index out of bounds: index = {index} but the length was {}",
					tuple.len()
				))
			})?;
			Ok(value.clone())
		}),
	);
	ctx.set_function(
		istr("get"),
		evalexpr::RustFunction::new(|s, _| {
			expect_function_argument_amount(s.num_args(), 2)?;
			let tuple = s.get_arg(0).unwrap().as_tuple_ref()?;
			let index: T = s.get_arg(1).unwrap().as_float()?;

			let index = index.to_f64() as usize;

			let value = tuple.get(index).ok_or_else(|| {
				EvalexprError::CustomMessage(format!(
					"Index out of bounds: index = {index} but the length was {}",
					tuple.len()
				))
			})?;
			Ok(value.clone())
		}),
	);
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
	dragging_point_i:     Option<draw_buffer::PointInteraction>,
	plot_mouese_pos:      Option<(egui::Pos2, PlotTransform)>,
	showing_custom_label: bool,

	f32_epsilon:          f64,
	f64_epsilon:          f64,
	permalink_string:     String,
	scheduled_url_update: bool,
	last_url_update:      f64,
	file_to_remove:       Option<String>,

	draw_buffers: Box<ThreadLocal<DrawBufferRC>>,
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
				draw_buffers: Box::new(ThreadLocal::new()),
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
			app_ui::side_panel(&mut self.state_f32, &mut self.ui, ctx, frame);
			app_ui::graph_panel(&mut self.state_f32, &mut self.ui, ctx);
		} else {
			app_ui::side_panel(&mut self.state_f64, &mut self.ui, ctx, frame);
			app_ui::graph_panel(&mut self.state_f64, &mut self.ui, ctx);
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

#[cold]
fn cold() {}
pub fn unlikely(x: bool) -> bool {
	if x {
		cold()
	}
	x
}
pub fn thread_local_get<T: Send + Default>(tl: &ThreadLocal<T>) -> &T {
	if let Some(t) = tl.get() {
		t
	} else {
		cold();
		tl.get_or_default()
	}
}
