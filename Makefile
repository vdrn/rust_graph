all:
	wasm-pack build --release --target web --out-dir assets/wasm --config profile.release.codegen-units=1
	rm assets/wasm/.gitignore

