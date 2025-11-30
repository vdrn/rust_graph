#![feature(adt_const_params)]
extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::cell::RefCell;
use std::env;

use eframe::egui::{self, Id, Visuals};
use eframe::{App, CreationContext};
use egui_plot::PlotBounds;
use evalexpr::{DefaultNumericTypes, EvalexprFloat, F32NumericTypes, Stack};
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
use custom_rendering::CustomRenderer;
use draw_buffer::DrawBufferRC;
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

use crate::draw_buffer::{ProcessedShape, ProcessedShapes};

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

struct State<T: EvalexprFloat> {
	entries:              Vec<Entry<T>>,
	ctx:                  evalexpr::HashMapContext<T>,
	saved_graph_config:   GraphConfig,
	name:                 String,
	clear_cache:          bool,
	thread_local_context: Arc<ThreadLocal<ThreadLocalContext<T>>>,
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
	processed_shapess: ProcessedShapes,
	conf:              AppConfig,
	next_id:           u64,
	plot_bounds:       PlotBounds,
	graph_config:      GraphConfig,

	// data_aspect: f32,
	reset_graph:            bool,
	force_process_elements: bool,
	force_create_elements:  bool,

	cur_dir:             String,
	serialization_error: Option<String>,
	serialized_states:   BTreeMap<String, String>,

	parsing_errors:      FxHashMap<u64, String>,
	prepare_errors:      FxHashMap<u64, String>,
	optimization_errors: FxHashMap<u64, String>,
	eval_errors:         FxHashMap<u64, String>,

	selected_plot_line:   Option<(Id, bool)>,
	dragging_point_i:     Option<draw_buffer::PointInteraction>,
	plot_mouese_pos:      Option<egui::Pos2>,
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

	custom_renderer:     Option<CustomRenderer>,
	prev_plot_transform: Option<egui_plot::PlotTransform>,
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
		// let mut fonts = egui::FontDefinitions::default();
		// egui_nerdfonts::add_to_fonts(&mut fonts, egui_nerdfonts::Variant::Regular);
		// cc.egui_ctx.set_fonts(fonts);

		let mut entries_s = Vec::new();
		let mut entries_d = Vec::new();

		// TODO
		let mut serialized_states = BTreeMap::default();

		let mut serialization_error = None;

		let conf: AppConfig = cc
			.storage
			.and_then(|s| s.get_string(CONF_KEY).and_then(|d| serde_json::from_str(&d).ok()))
			.unwrap_or_default();

		let mut default_graph_config_s = conf.default_graph_config.clone();
		let mut default_graph_config_d = conf.default_graph_config.clone();
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

		if entries_s.is_empty() {
			next_id += 1;
			entries_s.push(Entry::new_function(0, "sin(x)"));
		}
		let ctx_s = evalexpr::HashMapContext::new();

		if entries_d.is_empty() {
			next_id += 1;
			entries_d.push(Entry::new_function(0, "sin(x)"));
		}
		let ctx_d = evalexpr::HashMapContext::new();

		Self {
			#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
			_puffin: run_puffin_server(),
			state_f32: State {
				entries:              entries_s,
				ctx:                  ctx_s,
				name:                 String::new(),
				saved_graph_config:   default_graph_config_s.clone(),
				// points_cache: PointsCache::default(),
				clear_cache:          true,
				thread_local_context: Arc::new(ThreadLocal::new()),
			},
			state_f64: State {
				entries:              entries_d,
				saved_graph_config:   default_graph_config_d,
				ctx:                  ctx_d,
				name:                 String::new(),
				// points_cache: PointsCache::default(),
				clear_cache:          true,
				thread_local_context: Arc::new(ThreadLocal::new()),
			},
			ui: UiState {
				force_process_elements: true,
				force_create_elements: true,
				processed_shapess: ProcessedShapes::new(),
				custom_renderer: CustomRenderer::new(cc),
				#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
				full_frame_scope: None,
				// animating: Arc::new(AtomicBool::new(true)),
				conf,
				scheduled_url_update: false,
				last_url_update: 0.0,
				serialized_states,
				next_id,
				plot_bounds: PlotBounds::from_min_max([-2.0, -2.0], [2.0, 2.0]),
				graph_config: default_graph_config_s,
				reset_graph: false,

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
				f32_epsilon: f32::EPSILON as f64,
				f64_epsilon: f64::EPSILON,
				permalink_string: String::new(),
				file_to_remove: None,
				draw_buffers: Box::new(ThreadLocal::new()),
				prev_plot_transform: None,
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
			let changed = app_ui::side_panel(&mut self.state_f32, &mut self.ui, ctx, frame);
			app_ui::graph_panel(&mut self.state_f32, &mut self.ui, ctx, frame, changed);
		} else {
			let changed = app_ui::side_panel(&mut self.state_f64, &mut self.ui, ctx, frame);
			app_ui::graph_panel(&mut self.state_f64, &mut self.ui, ctx, frame, changed);
		}
		if use_f32 != self.ui.conf.use_f32 {
			if use_f32 {
				let mut output = Vec::with_capacity(1024);
				if persistence::serialize_to_json(
					&mut output,
					&self.state_f32.entries,
					self.ui.graph_config.clone(),
				)
				.is_ok()
				{
					let (entries, default_graph_config) =
						persistence::deserialize_from_json(&output, &mut self.ui.next_id).unwrap();
					self.state_f64.entries = entries;
					self.state_f64.saved_graph_config = default_graph_config;
					self.state_f64.clear_cache = true;
					self.state_f64.name = self.state_f32.name.clone();
				}
			} else {
				let mut output = Vec::with_capacity(1024);
				if persistence::serialize_to_json(
					&mut output,
					&self.state_f64.entries,
					self.ui.graph_config.clone(),
				)
				.is_ok()
				{
					let (entries, default_graph_config) =
						persistence::deserialize_from_json(&output, &mut self.ui.next_id).unwrap();
					self.state_f32.entries = entries;
					self.state_f32.saved_graph_config = default_graph_config;
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
