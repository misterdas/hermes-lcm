#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
if [[ -n "${HERMES_PROFILE:-}" ]]; then
  TARGET_ROOT="$HERMES_HOME_DIR/profiles/${HERMES_PROFILE}"
else
  TARGET_ROOT="$HERMES_HOME_DIR"
fi

PLUGIN_TARGET="$TARGET_ROOT/plugins/hermes-trove"
SKILL_TARGET="$TARGET_ROOT/skills/hermes-trove"
CONFIG="$TARGET_ROOT/config.yaml"
ENV_FILE="$TARGET_ROOT/.env"

removed=false
foreign_checkout=false

# Remove plugin symlink
if [[ -L "$PLUGIN_TARGET" ]]; then
  target="$(readlink "$PLUGIN_TARGET")"
  if [[ "$target" == "$REPO_ROOT" ]]; then
    rm "$PLUGIN_TARGET"
    echo "Removed plugin symlink: $PLUGIN_TARGET"
    removed=true
  else
    foreign_checkout=true
    echo "Skipping plugin symlink (points to $target, not $REPO_ROOT)" >&2
  fi
elif [[ -e "$PLUGIN_TARGET" ]]; then
  echo "Skipping plugin (not a symlink): $PLUGIN_TARGET" >&2
fi

# Remove skill symlink
if [[ -L "$SKILL_TARGET" ]]; then
  target="$(readlink "$SKILL_TARGET")"
  expected="$REPO_ROOT/skills/hermes-trove"
  if [[ "$target" == "$expected" ]]; then
    rm "$SKILL_TARGET"
    echo "Removed skill symlink: $SKILL_TARGET"
    removed=true
  else
    foreign_checkout=true
    echo "Skipping skill symlink (points to $target, not $expected)" >&2
  fi
elif [[ -e "$SKILL_TARGET" ]]; then
  echo "Skipping skill (not a symlink): $SKILL_TARGET" >&2
fi

# Remove hermes-trove sections from config.yaml
if [[ -f "$CONFIG" && "$foreign_checkout" == "false" ]]; then
  if python3 -c 'import yaml' >/dev/null 2>&1; then
    # Use Python to properly remove the YAML sections
    python3 -c "
import yaml, sys

with open('$CONFIG') as f:
    data = yaml.safe_load(f)

changed = False
if 'plugins' in data and 'enabled' in data['plugins']:
    orig = data['plugins']['enabled'][:]
    data['plugins']['enabled'] = [p for p in orig if p != 'hermes-trove']
    if not data['plugins']['enabled']:
        del data['plugins']['enabled']
    if not data['plugins']:
        del data['plugins']
    changed = True

if 'context' in data and data.get('context', {}).get('engine') == 'trove':
    del data['context']['engine']
    if not data['context']:
        del data['context']
    changed = True

if changed:
    with open('$CONFIG', 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print('Removed hermes-trove from config.yaml')
else:
    print('config.yaml already clean')
"
  else
    echo "config cleanup requires PyYAML; left $CONFIG unchanged"
  fi
fi

# Remove Trove-owned environment defaults
if [[ -f "$ENV_FILE" && "$foreign_checkout" == "false" ]]; then
  env_tmp="${ENV_FILE}.trovetmp"
  awk '
    /^TROVE_ENABLE_SLASH_COMMAND=/ { next }
    /^TROVE_RETENTION_DAYS=/ { next }
    { print }
  ' "$ENV_FILE" > "$env_tmp"
  mv "$env_tmp" "$ENV_FILE"
  echo "Removed Trove settings from $ENV_FILE"
fi

if [[ "$removed" == "false" ]]; then
  echo "Nothing to uninstall — hermes-trove not found"
else
  echo "Uninstalled hermes-trove"
fi
