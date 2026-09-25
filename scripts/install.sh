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
SKILL_SOURCE="$REPO_ROOT/skills/hermes-trove"
SKILL_TARGET="$TARGET_ROOT/skills/hermes-trove"

preflight_target() {
  local label="$1"
  local target="$2"
  local expected="$3"

  if [[ -L "$target" ]]; then
    local current_target
    current_target="$(readlink "$target")"
    if [[ "$current_target" != "$expected" ]]; then
      if [[ "$label" == "plugin" ]]; then
        echo "Refusing to replace existing symlink: $target -> $current_target" >&2
      else
        echo "Refusing to replace existing skill symlink: $target -> $current_target" >&2
      fi
      echo "Remove it manually or point it at this checkout before rerunning install.sh." >&2
      exit 1
    fi
  elif [[ -e "$target" ]]; then
    if [[ "$label" == "plugin" && -d "$target" ]]; then
      local physical_target
      physical_target="$(cd "$target" && pwd -P)"
      if [[ "$physical_target" == "$expected" ]]; then
        return
      fi
    fi
    if [[ "$label" == "plugin" ]]; then
      echo "Refusing to replace existing path: $target" >&2
    else
      echo "Refusing to replace existing skill path: $target" >&2
    fi
    echo "Move it aside or remove it manually before rerunning install.sh." >&2
    exit 1
  fi
}

preflight_target "plugin" "$PLUGIN_TARGET" "$REPO_ROOT"
preflight_target "skill" "$SKILL_TARGET" "$SKILL_SOURCE"

mkdir -p "$(dirname "$PLUGIN_TARGET")" "$(dirname "$SKILL_TARGET")"

if [[ ! -e "$PLUGIN_TARGET" && ! -L "$PLUGIN_TARGET" ]]; then
  ln -s "$REPO_ROOT" "$PLUGIN_TARGET"
fi
if [[ ! -e "$SKILL_TARGET" && ! -L "$SKILL_TARGET" ]]; then
  ln -s "$SKILL_SOURCE" "$SKILL_TARGET"
fi

cat <<EOF
Installed hermes-trove at:
  $PLUGIN_TARGET

Discoverable skill:
  $SKILL_TARGET

Activation requires both:

plugins:
  enabled:
    - hermes-trove

context:
  engine: trove

Verification:
  1. Restart Hermes.
  2. Run: hermes plugins
  3. Confirm the plugin list includes hermes-trove and the selected context engine is trove.
  4. Confirm the available skills include hermes-trove.
EOF

# Auto-configure config.yaml if needed. Use a YAML-aware merge so we never
# append a duplicate top-level `plugins:` / `context:` block (duplicate keys
# are invalid YAML — PyYAML silently drops the earlier one, which can wipe
# out the user's existing plugin list).
CONFIG="$TARGET_ROOT/config.yaml"
if [[ -f "$CONFIG" ]]; then
  if python3 - "$CONFIG" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
try:
    import yaml
except ImportError:
    sys.exit(2)          # no PyYAML -> caller falls back to manual block
data = yaml.safe_load(p.read_text()) or {}
if not isinstance(data, dict):
    sys.exit(1)          # unexpected top-level type -> caller aborts
plugins = data.setdefault("plugins", {})
enabled = plugins.setdefault("enabled", [])
if "hermes-trove" not in enabled:
    enabled.append("hermes-trove")
context = data.setdefault("context", {})
if context.get("engine") in (None, "", "default"):
    context["engine"] = "trove"
tmp = p.with_suffix(".yaml.trovetmp")
tmp.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
tmp.replace(p)
PY
    then
      echo "Auto-configured $CONFIG"
    else
      # PyYAML missing, or unexpected config shape: never blind-append, tell the user.
      echo "Could not auto-merge $CONFIG — add manually:"
      echo "  plugins:"
      echo "    enabled:"
      echo "      - hermes-trove"
      echo "  context:"
      echo "    engine: trove"
    fi
else
  echo "No config.yaml found at $CONFIG — add manually:"
  echo "  plugins:"
  echo "    enabled:"
  echo "      - hermes-trove"
  echo "  context:"
  echo "    engine: trove"
fi

# Enable slash commands
ENV_FILE="$TARGET_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
  if [[ -s "$ENV_FILE" && "$(tail -c 1 "$ENV_FILE" | wc -l)" -eq 0 ]]; then
    printf '\n' >> "$ENV_FILE"
  fi
  if awk -F= '/^[[:space:]]*TROVE_ENABLE_SLASH_COMMAND=/ { found=1 } END { exit !found }' "$ENV_FILE"; then
    echo "Slash commands already configured in $ENV_FILE"
  else
    echo "TROVE_ENABLE_SLASH_COMMAND=1" >> "$ENV_FILE"
    echo "Added TROVE_ENABLE_SLASH_COMMAND=1 to $ENV_FILE"
  fi
  if awk -F= '/^[[:space:]]*TROVE_RETENTION_DAYS=/ { found=1 } END { exit !found }' "$ENV_FILE"; then
    echo "TROVE_RETENTION_DAYS already configured in $ENV_FILE"
  else
    echo "TROVE_RETENTION_DAYS=0" >> "$ENV_FILE"
    echo "Added TROVE_RETENTION_DAYS=0 to $ENV_FILE (0 = keep raw messages forever; set e.g. 90 to auto-delete sessions older than 90 days)"
  fi
else
  {
    echo "TROVE_ENABLE_SLASH_COMMAND=1"
    echo "TROVE_RETENTION_DAYS=0"
  } > "$ENV_FILE"
  echo "Created $ENV_FILE with TROVE_ENABLE_SLASH_COMMAND=1 and TROVE_RETENTION_DAYS=0"
fi
