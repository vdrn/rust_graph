# Rust Graph
Desmos-like graphing calculator with support for f32 numbers.

[Web Demo](https://vdrn.github.io/rust_graph/#%5B%7B%22name%22%3A%22c1%22%2C%22visible%22%3Atrue%2C%22color%22%3A1%2C%22value%22%3A%7B%22Constant%22%3A%7B%22value%22%3A2.5692599999999657%2C%22step%22%3A0.01744%2C%22ty%22%3A%7B%22LoopForwardAndBackward%22%3A%7B%22start%22%3A0.1%2C%22end%22%3A4.87%2C%22forward%22%3Afalse%7D%7D%7D%7D%7D%2C%7B%22name%22%3A%22f%22%2C%22visible%22%3Atrue%2C%22color%22%3A2%2C%22value%22%3A%7B%22Function%22%3A%22sin%28x%5E3%20%2B%20c1%20%2A%20c1%29%22%7D%7D%2C%7B%22name%22%3A%22%22%2C%22visible%22%3Atrue%2C%22color%22%3A4%2C%22value%22%3A%7B%22Integral%22%3A%7B%22func_text%22%3A%22f%28x%29%2A0.5%22%2C%22lower_text%22%3A%22-c1%22%2C%22upper_text%22%3A%22c1%22%2C%22resolution%22%3A500%7D%7D%7D%2C%7B%22name%22%3A%22c3%22%2C%22visible%22%3Atrue%2C%22color%22%3A6%2C%22value%22%3A%7B%22Constant%22%3A%7B%22value%22%3A2.399999999999993%2C%22step%22%3A0.01%2C%22ty%22%3A%7B%22LoopForward%22%3A%7B%22start%22%3A0.0%2C%22end%22%3A3.14159265%7D%7D%7D%7D%7D%2C%7B%22name%22%3A%22%22%2C%22visible%22%3Atrue%2C%22color%22%3A7%2C%22value%22%3A%7B%22Points%22%3A%5B%7B%22x%22%3A%220%22%2C%22y%22%3A%220%22%7D%2C%7B%22x%22%3A%22cos%28c3%2A2%29%2A5%22%2C%22y%22%3A%22sin%28c3%2A2%29%2A5%22%7D%5D%7D%7D%5D)

## Compile and install locally

Make sure you have recent [Rust compiler installed](https://rust-lang.org/tools/install/).

Then run:
```bash
cargo install --git https://github.com/vdrn/rust_graph
```
