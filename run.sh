#!/bin/bash

clean() {
    echo "Cleaning up..."
}

build() {
    echo "Building the project..."
    ./cleanfile.sh
    cmake ./project/ && make -j8    
    ./cleanfile.sh
}


case "$1" in
    clean) clean ;;
    build) build ;;
    *) echo "Usage: $0 {clean|build}" ;;
esac
