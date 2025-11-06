use rust_graph::Application;

fn main() {
	// #[cfg(target_arch = "wasm32")]
	// {
	// 	rust_graph::wasm_main("");
	// }
	#[cfg(not(target_arch = "wasm32"))]
	{
		let mut opts = eframe::NativeOptions::default();
		opts.viewport =
			opts.viewport.with_fullscreen(true).with_inner_size(eframe::egui::Vec2::new(1000.0, 600.0));
		eframe::run_native("Rust Graph", opts, Box::new(|cc| Ok(Box::new(Application::new(cc))))).unwrap();
	}
}
