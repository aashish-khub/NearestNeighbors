# Releasing a New Version of $N^2$

This project uses GitHub Actions and trusted publishing to automatically deploy to PyPI when a GitHub release is created.

## Steps to Release

1. **Update the version** in `pyproject.toml`:

   For example:
   ```toml
   version = "1.x.y"
   ```
> [!NOTE]
> Follow [Semantic Versioning](https://semver.org/) for versioning.
2. **Commit and push** the change:
   ```bash
   git add pyproject.toml
   git commit -m "Update version to 1.x.y"
   git push origin <your_working_branch>
   ```
   Then seek approval for the pull request and merge it into `main` as usual.

3. **Create a new Git tag and push it**:
   ```bash
   git tag v1.x.y
   git push origin v1.x.y
   ```

4. **Draft a new GitHub Release**:
   - Go to the [Releases tab](https://github.com/aashish-khub/NearestNeighbors/releases)
   - Click **"Draft a new release"**
   - Set the tag to `v1.x.y`.
   - Include release notes in the description.
   - **Do not check "prerelease"** if you intend to publish to PyPI
   - Click **"Publish release"**

## Notes

- This triggers `.github/workflows/publish.yml`
- Prereleases (e.g., beta versions) are published to [Test PyPI](https://test.pypi.org/project/nsquared/) (i.e. a dev pypi environment)
- Final releases go to [PyPI](https://pypi.org/project/nsquared/)

## Troubleshooting

- Make sure the version in `pyproject.toml` exactly matches the tag.
- Check Actions tab for workflow logs if publishing fails.
