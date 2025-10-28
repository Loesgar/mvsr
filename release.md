## Creating a new release

1. Ensure all versions have been bumped.

   Currently the version is specified in:
   - [`mvsr/CMakeLists.txt`](mvsr/CMakeLists.txt)
   - [`lang/python/pyproject.toml`](lang/python/pyproject.toml)

   Additionally any lock files should also be updated (this is technically not necessary though):
   - run `uv lock` in [`lang/python`](lang/python)
   - run `uv lock` in [`docs`](docs)

2. Push all your changes to the `main` branch and ensure the CI pipelines are passing.

3. [Draft a new release](https://github.com/Loesgar/mvsr/releases/new) using the GitHub interface.
   Create a new tag called `v<major>.<minor>.<patch>` (for example `v1.0.0`).
   Use the same name for the name of the release.

4. Publish the release and ensure the release workflow passes.
   If it fails, fix any issues, delete the release and repeat step 3 again.

5. Look through all the commits since the previous release and add appropriate patch notes to the new release.
