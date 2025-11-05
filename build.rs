#[cfg(target_os = "windows")]
extern crate winres;

fn main() {
	#[cfg(target_os = "windows")]
	{
		let mut res = winres::WindowsResource::new();
		res.set_icon("assets\\favicon.ico");
		res.compile().unwrap();
	}
}
