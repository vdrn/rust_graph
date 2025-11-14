all:
	wasm-pack build --target web --out-dir assets/wasm --config profile.release.codegen-units=1
	rm assets/wasm/.gitignore

serve:
	serve --config ./serve.json ./assets

bp:
	cargo build --profile profiling
