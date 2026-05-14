#![feature(adt_const_params)]
extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::env;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use eframe::egui::{self, Id, Visuals};
#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::{self, prelude::*};
use eframe::{App, CreationContext};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use thread_local::ThreadLocal;
#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

use evalexpr::{DefaultNumericTypes, EvalexprFloat, F32NumericTypes, HashMapContext};

mod builtins;
mod color;
mod custom_rendering;
mod entry;
mod graph_ui;
mod marching_squares;
mod math_utils;
mod persistence;
mod plot_extensions;
mod side_panel_ui;
mod utils;

use color::ProcessedColors;
use custom_rendering::fan_fill_renderer::FanFillRenderer;
use custom_rendering::mesh_renderer::init_mesh_renderer;
use entry::{ConstantType, Entry, EntryType, PointEntry};
use graph_ui::create_plot_elements::ThreadLocalContext;
use graph_ui::graph_config::GraphConfig;
use graph_ui::plot_elements::{PlotElementsRC, ProcessedPlotElements};
use graph_ui::plot_elements_scheduler::PlotElementsScheduler;
use graph_ui::{DebugInfo, graph_ui};

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

use crate::graph_ui::GraphState;

#[derive(Clone, PartialEq)]
enum PendingAction {
	New,
	Open,
	Close,
}

fn has_unsaved_changes(ui_state: &UiState) -> bool {
	ui_state.initial_permalink_string.as_ref() != Some(&ui_state.permalink_string)
}

const DATA_KEY: &str = "data";
const CONF_KEY: &str = "conf";
static EXAMPLES_DIR: include_dir::Dir<'_> = include_dir::include_dir!("$CARGO_MANIFEST_DIR/examples");

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
pub fn wasm_main() -> () {
	use eframe::egui_wgpu::wgpu::Backends;
	use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup, WgpuSetupCreateNew};
	use eframe::wasm_bindgen::JsCast as _;

	let mut web_options = eframe::WebOptions::default();

	web_options.wgpu_options = WgpuConfiguration {
		wgpu_setup: WgpuSetup::CreateNew(WgpuSetupCreateNew {
			instance_descriptor: eframe::wgpu::InstanceDescriptor { backends: Backends::GL, ..eframe::wgpu::InstanceDescriptor::new_without_display_handle() },

      ..WgpuSetupCreateNew::without_display_handle()
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
	app_config: AppConfig,

	// UI
	showing_help:        bool,
	showing_settings:    bool,
	showing_open:        bool,
	showing_save:        bool,
	showing_save_prompt: bool,

	// UI - Persistance
	serialized_states: BTreeMap<String, String>,
	#[cfg(target_arch = "wasm32")]
	file_to_remove:    Option<String>,
	#[cfg(not(target_arch = "wasm32"))]
	file_dialog:       egui_file_dialog::FileDialog,

	// UI: Errors
	serialization_error: Option<String>,
	parsing_errors:      FxHashMap<u64, String>,
	prepare_errors:      FxHashMap<u64, String>,
	optimization_errors: FxHashMap<u64, String>,
	eval_errors:         FxHashMap<u64, String>,

	// UI: Graph interactions
	selected_plot_line:   Option<(Id, bool)>,
	dragging_point_i:     Option<graph_ui::plot_elements::PointInteraction>,
	plot_mouese_pos:      Option<egui::Pos2>,
	showing_custom_label: bool,

	// URL
	permalink_string:     String,
	scheduled_url_update: bool,
	last_url_update:      f64,

	// File state
	current_file_path:          Option<String>,
	initial_permalink_string:   Option<String>,
	pending_action:             Option<PendingAction>,
	perform_pending_after_save: bool,

	// cross frame signals
	reset_graph:            bool,
	force_process_elements: bool,
	clear_cache:            bool,

	// Drawing
	processed_shapes:            ProcessedPlotElements,
	draw_buffers:                Box<ThreadLocal<PlotElementsRC>>,
	fan_fill_renderer:           FanFillRenderer,
	multi_draw_buffer_scheduler: PlotElementsScheduler,

	// Misc
	debug_info: DebugInfo,
}
impl UiState {
	pub fn quit(&mut self, ctx: &egui::Context) {
		if crate::has_unsaved_changes(self) {
			use crate::PendingAction;

			self.showing_save_prompt = true;
			self.pending_action = Some(PendingAction::Close);
		} else {
			ctx.send_viewport_cmd(egui::ViewportCommand::Close);
		}
	}
}

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
fn run_puffin_server() -> puffin_http::Server {
	let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
	let puffin_server = puffin_http::Server::new(&server_addr).unwrap();
	println!("Run this to view profiling data:  puffin_viewer {server_addr}");

	puffin::set_scopes_on(true);

	puffin_server
}
impl Application {
	#[allow(unused_mut)]
	pub fn new(cc: &CreationContext) -> Self {
		let conf: AppConfig = cc
			.storage
			.and_then(|s| s.get_string(CONF_KEY).and_then(|d| serde_json::from_str(&d).ok()))
			.unwrap_or_default();
		if !conf.fullscreen {
			cc.egui_ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
		}

		init_mesh_renderer(cc);

		let mut serialized_states = BTreeMap::default();
		let mut serialization_error = None;
		let mut graph_state_s = GraphState::new(conf.default_graph_config.clone());
		let mut graph_state_d = GraphState::new(conf.default_graph_config.clone());
		let cur_dir;
		#[cfg(target_arch = "wasm32")]
		{
			cur_dir = String::new();
			// 	let error: Option<String> = web::get_data_from_url(&mut data);
			if let Some(storage) = cc.storage {
				if let Some(data) = storage.get_string(DATA_KEY) {
					serialized_states = serde_json::from_str(&data).unwrap();
				}
			}

			match persistence::deserialize_from_url::<F32NumericTypes>() {
				Ok(data) => {
					if let Some(data) = data {
						graph_state_s = data;
					}
				},
				Err(e) => {
					serialization_error = Some(e);
				},
			}
			match persistence::deserialize_from_url::<DefaultNumericTypes>() {
				Ok(data) => {
					if let Some(data) = data {
						graph_state_d = data;
					}
				},
				Err(e) => {
					serialization_error = Some(e);
				},
			}
		}

		#[cfg(not(target_arch = "wasm32"))]
		{
			cur_dir = env::home_dir()
				.and_then(|d| d.join("rust_graphs").to_str().map(|s| s.to_string()))
				.unwrap_or_default();
		}

		Self {
			#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
			_puffin: run_puffin_server(),
			#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
			full_frame_scope: None,
			state_f32: State::new(graph_state_s),
			state_f64: State::new(graph_state_d),
			ui: UiState {
				#[cfg(not(target_arch = "wasm32"))]
				file_dialog: egui_file_dialog::FileDialog::new()
					.initial_directory(PathBuf::from(&cur_dir))
					.add_save_extension("Rust Graph JSON file", "json")
					.default_save_extension("Rust Graph JSON file")
					.allow_path_edit_to_save_file_without_extension(false)
					.add_file_filter(
						"Rust Graph JSON file",
						egui_file_dialog::Filter::new(|p:&Path| p.extension().unwrap_or_default() == "json"),
					)
					.default_file_filter("Rust Graph JSON file"),
				#[cfg(target_arch = "wasm32")]
				file_to_remove: None,

				debug_info: DebugInfo::new(),
				multi_draw_buffer_scheduler: PlotElementsScheduler::new(),
				force_process_elements: true,
				processed_shapes: ProcessedPlotElements::new(),
				fan_fill_renderer: FanFillRenderer::new(cc),
				app_config: conf,
				scheduled_url_update: false,
				last_url_update: 0.0,
				serialized_states,
				reset_graph: false,
				clear_cache: true,

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
				current_file_path: None,
				initial_permalink_string: None,
				pending_action: None,
				perform_pending_after_save: false,
				draw_buffers: Box::new(ThreadLocal::new()),
				showing_help: false,
				showing_settings: false,
				showing_open: false,
				showing_save: false,
				showing_save_prompt: false,
			},
		}
	}
}
impl App for Application {
	fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
		#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
		{
			self.ui.full_frame_scope.take();
			puffin::GlobalProfiler::lock().new_frame();
			self.ui.full_frame_scope = puffin::profile_scope_custom!("full_frame");
		}

		if self.ui.app_config.dark_mode {
			ui.set_visuals(Visuals::dark());
		} else {
			ui.set_visuals(Visuals::light());
		}
		ui.set_pixels_per_point(self.ui.app_config.ui_scale);

		let use_f32 = self.ui.app_config.use_f32;

		if use_f32 {
			let changed = side_panel_ui::side_panel(&mut self.state_f32, &mut self.ui, ui, frame);
			graph_ui(ui, &mut self.state_f32, &mut self.ui, frame, changed);
		} else {
			let changed = side_panel_ui::side_panel(&mut self.state_f64, &mut self.ui, ui, frame);
			graph_ui(ui, &mut self.state_f64, &mut self.ui, frame, changed);
		}
		if use_f32 != self.ui.app_config.use_f32 {
			let current_file_path = self.ui.current_file_path.clone();
			if use_f32 {
				// convert f32 graph to f64
				match self.state_f32.graph_state.convert_float_type() {
					Ok(converted_graph_state) => {
						load_graph_state(
							&mut self.ui,
							&mut self.state_f64,
							Ok(converted_graph_state),
							current_file_path,
						);
					},
					Err(e) => {
						self.ui.serialization_error = Some(e);
					},
				}
			} else {
				// convert f64 graph to f32
				match self.state_f64.graph_state.convert_float_type() {
					Ok(converted_graph_state) => {
						load_graph_state(
							&mut self.ui,
							&mut self.state_f32,
							Ok(converted_graph_state),
							current_file_path,
						);
					},
					Err(e) => {
						self.ui.serialization_error = Some(e);
					},
				}
			}
		}
		if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
			self.ui.quit(ui);
		}
	}

	fn save(&mut self, storage: &mut dyn eframe::Storage) {
		storage.set_string(DATA_KEY, serde_json::to_string(&self.ui.serialized_states).unwrap());
		storage.set_string(CONF_KEY, serde_json::to_string(&self.ui.app_config).unwrap());
	}

	fn auto_save_interval(&self) -> core::time::Duration { core::time::Duration::from_secs(5) }

	fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
		egui::Color32::from_rgba_unmultiplied(12, 12, 12, 180).to_normalized_gamma_f32()
	}

	fn persist_egui_memory(&self) -> bool { true }

	fn raw_input_hook(&mut self, _ctx: &egui::Context, _raw_input: &mut egui::RawInput) {}
}

fn load_graph_state<T: EvalexprFloat>(
	ui_state: &mut UiState, state: &mut State<T>, graph_state: Result<GraphState<T>, String>,
	file_path: Option<String>,
) {
	match graph_state {
		Ok(graph_state) => {
			state.graph_state = graph_state;
			ui_state.serialization_error = None;
			ui_state.clear_cache = true;
			ui_state.reset_graph = true;
			ui_state.multi_draw_buffer_scheduler = PlotElementsScheduler::new();
			ui_state.current_file_path = file_path;
			ui_state.initial_permalink_string = None; // will be set when permalink is updated
		},
		Err(e) => {
			ui_state.serialization_error = Some(e);
		},
	}
}
