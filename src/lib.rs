#![feature(adt_const_params)]
extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::cell::RefCell;
use std::env;

use eframe::egui::{self, Id, Visuals};
use eframe::{App, CreationContext};
use egui_plot::PlotBounds;
use evalexpr::{DefaultNumericTypes, EvalexprFloat, F32NumericTypes, HashMapContext, Stack};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use thread_local::ThreadLocal;

mod app_ui;
mod builtins;
mod custom_rendering;
mod draw_buffer;
mod entry;
mod marching_squares;
mod math;
mod persistence;
mod widgets;

use app_ui::GraphConfig;
use custom_rendering::fan_fill_renderer::FanFillRenderer;
use custom_rendering::mesh_renderer::init_mesh_renderer;
use draw_buffer::{DrawBufferRC, MultiDrawBufferScheduler, ProcessedShapes};
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

use crate::app_ui::DebugInfo;
use crate::entry::ProcessedColors;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_main() -> () {
	use eframe::egui_wgpu::wgpu::Backends;
	use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup, WgpuSetupCreateNew};
	use eframe::wasm_bindgen::JsCast as _;

	let mut web_options = eframe::WebOptions::default();

	web_options.wgpu_options = WgpuConfiguration {
		wgpu_setup: WgpuSetup::CreateNew(WgpuSetupCreateNew {
			instance_descriptor: wgpu::InstanceDescriptor { backends: Backends::GL, ..Default::default() },
			..Default::default()
		}),
		..Default::default()
	};

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
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct IdGenerator {
	next_id: u64,
}
impl IdGenerator {
	fn next(&mut self) -> u64 {
		self.next_id += 1;
		self.next_id
	}
	pub fn new(start_id: u64) -> Self { Self { next_id: start_id } }
}

struct GraphState<T: EvalexprFloat> {
	pub entries:              Vec<Entry<T>>,
	pub saved_graph_config:   GraphConfig,
	pub current_graph_config: GraphConfig,
	pub name:                 String,
	pub id_gen:               IdGenerator,
	pub prev_plot_transform:  Option<egui_plot::PlotTransform>,
}
impl<T: EvalexprFloat> GraphState<T> {
	pub fn new(default_graph_config: GraphConfig) -> Self {
		let mut id_generator = IdGenerator::default();
		let first_id = id_generator.next();
		Self {
			entries:              vec![Entry::new_function(first_id, "sin(x)")],
			saved_graph_config:   default_graph_config.clone(),
			current_graph_config: default_graph_config,
			name:                 String::new(),
			id_gen:               id_generator,
			prev_plot_transform:  None,
		}
	}
	pub fn prev_plot_bounds(&self) -> PlotBounds {
		self.prev_plot_transform
			.as_ref()
			.map(|t| *t.bounds())
			.unwrap_or_else(|| PlotBounds::from_min_max([-2.0, -2.0], [2.0, 2.0]))
	}
}
struct State<T: EvalexprFloat> {
	graph_state:          GraphState<T>,
	ctx:                  evalexpr::HashMapContext<T>,
	thread_local_context: Arc<ThreadLocal<ThreadLocalContext<T>>>,
	processed_colors:     ProcessedColors<T>,
}
impl<T: EvalexprFloat> State<T> {
	pub fn new(graph_state: GraphState<T>) -> Self {
		Self {
			graph_state,
			ctx: HashMapContext::new(),
			thread_local_context: Arc::new(ThreadLocal::new()),
			processed_colors: ProcessedColors::new(),
		}
	}
}

// struct State<T: EvalexprFloat> {
// 	entries:              Vec<Entry<T>>,
// 	ctx:                  evalexpr::HashMapContext<T>,
// 	saved_graph_config:   GraphConfig,
// 	name:                 String,
// 	clear_cache:          bool,
// 	thread_local_context: Arc<ThreadLocal<ThreadLocalContext<T>>>,
// 	processed_colors:     ProcessedColors<T>,
// }

#[derive(Serialize, Deserialize)]
struct AppConfig {
	dark_mode:  bool,
	use_f32:    bool,
	resolution: usize,
	fullscreen: bool,

	ui_scale:             f32,
	default_graph_config: GraphConfig,
}
impl Default for AppConfig {
	fn default() -> Self {
		Self {
			fullscreen:           true,
			dark_mode:            true,
			use_f32:              false,
			resolution:           2000,
			ui_scale:             1.5,
			default_graph_config: GraphConfig::default(),
		}
	}
}

struct UiState {
	conf: AppConfig,

	// UI
	showing_help:      bool,
	// UI - Persistance
	cur_dir:           String,
	serialized_states: BTreeMap<String, String>,
	file_to_remove:    Option<String>,

	// UI: Errors
	serialization_error: Option<String>,
	parsing_errors:      FxHashMap<u64, String>,
	prepare_errors:      FxHashMap<u64, String>,
	optimization_errors: FxHashMap<u64, String>,
	eval_errors:         FxHashMap<u64, String>,

	// UI: Graph interactions
	selected_plot_line:   Option<(Id, bool)>,
	dragging_point_i:     Option<draw_buffer::PointInteraction>,
	plot_mouese_pos:      Option<egui::Pos2>,
	showing_custom_label: bool,

	// URL
	permalink_string:     String,
	scheduled_url_update: bool,
	last_url_update:      f64,

	// cross frame signals
	reset_graph:            bool,
	force_process_elements: bool,
	clear_cache:            bool,

	// Drawing
	processed_shapes:            ProcessedShapes,
	draw_buffers:                Box<ThreadLocal<DrawBufferRC>>,
	fan_fill_renderer:           Option<FanFillRenderer>,
	multi_draw_buffer_scheduler: MultiDrawBufferScheduler,

	// Misc
	debug_info: DebugInfo,
}
// #[derive(Clone, Debug)]
pub struct Application {
	state_f64:        State<DefaultNumericTypes>,
	state_f32:        State<F32NumericTypes>,
	ui:               UiState,
	#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
	_puffin:          puffin_http::Server,
	#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
	full_frame_scope: Option<puffin::ProfilerScope>,
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
		let mut serialized_states = BTreeMap::default();
		let mut serialization_error = None;

		let conf: AppConfig = cc
			.storage
			.and_then(|s| s.get_string(CONF_KEY).and_then(|d| serde_json::from_str(&d).ok()))
			.unwrap_or_default();

		let mut graph_state_s = GraphState::new(conf.default_graph_config.clone());
		let mut graph_state_d = GraphState::new(conf.default_graph_config.clone());

		if !conf.fullscreen {
			cc.egui_ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
		}

		init_mesh_renderer(cc);

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
            default_graph_config_s = bounds;

          },
          Err(e)=>{
            serialization_error = Some(e);
          }
        }
        match persistence::deserialize_from_url::<DefaultNumericTypes>(&mut next_id) {
          Ok((data,bounds))=>{
            entries_d = data;
            default_graph_config_d = bounds;
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

		Self {
			#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
			_puffin: run_puffin_server(),
			#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
			full_frame_scope: None,
			state_f32: State::new(graph_state_s),
			state_f64: State::new(graph_state_d),
			ui: UiState {
				debug_info: DebugInfo::new(),
				multi_draw_buffer_scheduler: MultiDrawBufferScheduler::new(),
				force_process_elements: true,
				processed_shapes: ProcessedShapes::new(),
				fan_fill_renderer: FanFillRenderer::new(cc),
				conf,
				scheduled_url_update: false,
				last_url_update: 0.0,
				serialized_states,
				reset_graph: false,
				clear_cache: true,

				cur_dir,
				serialization_error,
				parsing_errors: FxHashMap::default(),
				prepare_errors: FxHashMap::default(),
				optimization_errors: FxHashMap::default(),
				eval_errors: FxHashMap::default(),

				selected_plot_line: None,
				showing_custom_label: false,
				dragging_point_i: None,
				plot_mouese_pos: None,
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
		#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
		{
			self.ui.full_frame_scope.take();
			puffin::GlobalProfiler::lock().new_frame();
			self.ui.full_frame_scope = puffin::profile_scope_custom!("full_frame");
		}

		if self.ui.conf.dark_mode {
			ctx.set_visuals(Visuals::dark());
		} else {
			ctx.set_visuals(Visuals::light());
		}
		ctx.set_pixels_per_point(self.ui.conf.ui_scale);

		let use_f32 = self.ui.conf.use_f32;
		if use_f32 {
			let changed = app_ui::side_panel(&mut self.state_f32, &mut self.ui, ctx, frame);
			app_ui::graph_panel(&mut self.state_f32, &mut self.ui, ctx, frame, changed);
		} else {
			let changed = app_ui::side_panel(&mut self.state_f64, &mut self.ui, ctx, frame);
			app_ui::graph_panel(&mut self.state_f64, &mut self.ui, ctx, frame, changed);
		}
		if use_f32 != self.ui.conf.use_f32 {
			let mut output = Vec::with_capacity(1024);
			if use_f32 {
				let name = self.state_f32.graph_state.name.clone();
				if persistence::serialize_graph_state_to_json(&mut output, &self.state_f32.graph_state).is_ok()
				{
					let new_graph_state =
						persistence::deserialize_graph_state_from_json(name, &output).unwrap();
					load_graph_state(&mut self.ui, &mut self.state_f64, Ok(new_graph_state));
				}
			} else {
				let name = self.state_f64.graph_state.name.clone();
				if persistence::serialize_graph_state_to_json(&mut output, &self.state_f64.graph_state).is_ok()
				{
					let new_graph_state =
						persistence::deserialize_graph_state_from_json(name, &output).unwrap();
					load_graph_state(&mut self.ui, &mut self.state_f32, Ok(new_graph_state));
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

fn load_graph_state<T: EvalexprFloat>(
	ui_state: &mut UiState, state: &mut State<T>, graph_state: Result<GraphState<T>, String>,
) {
	match graph_state {
		Ok(graph_state) => {
			state.graph_state = graph_state;
			ui_state.serialization_error = None;
			ui_state.clear_cache = true;
			ui_state.reset_graph = true;
		},
		Err(e) => {
			ui_state.serialization_error = Some(e);
		},
	}
}
