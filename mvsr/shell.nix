{pkgs, ...}:
pkgs.mkShell {
    inputsFrom = [(pkgs.callPackage ./package.nix {})];
    packages = with pkgs; [clang-tools];
}
