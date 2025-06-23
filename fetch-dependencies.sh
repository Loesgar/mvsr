#!/bin/bash
PARENT_DIR=$(dirname "$(realpath -s "$0")")

git clone https://gitlab.com/libeigen/eigen.git "$PARENT_DIR/eigen"
(cd "$PARENT_DIR/eigen" && git checkout 2a9055b5)

git clone https://github.com/ludwigschmidt/fast-segmented-regression "$PARENT_DIR/related_work/fast_segmented_regression"
(cd "$PARENT_DIR/related_work/fast_segmented_regression" && git checkout 9d1c1c19)
