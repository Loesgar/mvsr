{ pkgs }:
pkgs.mkShell {
  packages = with pkgs; [
    python3
    uv
  ];

  LD_LIBRARY_PATH =
    with pkgs;
    lib.makeLibraryPath [
      stdenv.cc.cc.lib
      zlib
    ];
}
