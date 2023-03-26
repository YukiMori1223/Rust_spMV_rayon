FROM rust:latest

RUN apt update && apt upgrade -y
RUN apt install -y g++-mingw-w64-x86-64
RUN apt-get install -y gfortran

RUN rustup target add x86_64-pc-windows-gnu
RUN rustup toolchain install stable-x86_64-pc-windows-gnu
RUN rustup component add clippy
RUN rustup component add rustfmt

ENV CARGO_BUILD_TARGET_DIR=/tmp/target
