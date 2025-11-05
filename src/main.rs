use rust_graph::Application;

fn main() {
	// #[cfg(target_arch = "wasm32")]
	// {
	// 	rust_graph::wasm_main("");
	// }
	#[cfg(not(target_arch = "wasm32"))]
	{
		let mut opts = eframe::NativeOptions::default();
		opts.viewport = opts.viewport.with_fullscreen(true);
		eframe::run_native("graping", opts, Box::new(|cc| Ok(Box::new(Application::new(cc.storage)))))
			.unwrap();
	}
}

