{
  lib,
  stdenv,
  cmake,
  enableTests ? true,
}:
stdenv.mkDerivation {
  name = "mvsr";
  src = lib.sourceByRegex ./. [
    "^inc.*"
    "^src.*"
    "^test.*"
    "CMakeLists\.txt"
    # ".*\.pc\.in"
  ];
  nativeBuildInputs = [ cmake ];

  doCheck = true;
  cmakeFlags = lib.optionals (!enableTests) [ "-DTESTING=off" ];
}
