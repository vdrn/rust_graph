all:
	wasm-pack build --release --target web --out-dir assets/wasm
	rm assets/wasm/.gitignore

