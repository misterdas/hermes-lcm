# Release Procedure (maintainer)

Releasing a new version of `hermes-trove` follows this sequence. Each step is
verified before proceeding.

> This file is for maintainers who cut releases. End users and operators
> should not need to read it — it was moved out of the model-facing SKILL.md
> because agents were picking it up as instruction.

## Sequence

1. **Version bump**: Update `plugin.yaml` `version:` and all hardcoded
   version strings in tests (`test_trove_engine.py`,
   `test_trove_command.py`, `test_packaging_install.py`,
   `test_release_workflow.py`), `README.md`, `docs/operator-guide.md`,
   `CHANGELOG.md`, and `.github/ISSUE_TEMPLATE/bug_report.yml`.
   Use `grep -rln` to find all occurrences first.
2. **CHANGELOG restructure**: Move `## Unreleased` to become
   `## v{VERSION} - {DATE}`, then insert a fresh `## Unreleased` section
   at the top for future changes.
3. **Release notes**: Create `.github/release-notes/v{VERSION}.md` from
   the template. The release workflow at `.github/workflows/release.yml`
   requires this file to exist and start with `# hermes-trove v{VERSION}\n`.
4. **Commit and tag**:
   `git add -A && git commit -m "Release v{VERSION}" && git tag -a v{VERSION} -m "hermes-trove v{VERSION} release"`
5. **Push**: `git pull --rebase origin main` first (remote may have new
   commits), then `git push origin main && git push origin v{VERSION}`.
6. **Verify**: Run `pytest tests/test_release_workflow.py tests/test_trove_command.py -q`
   to confirm test consistency.

## Pitfalls

- The tag-driven CI marks tags containing `-` as prereleases and non-`-`
  tags as `latest`. A stable release must use a tag without `-`.
- `git reset --hard HEAD~1` destroys uncommitted work — commit before
  resetting.
- `.github/release-notes/v{VERSION}.md` must exist before pushing the
  tag, or the release workflow rejects it.
