#!/bin/bash
# Entrypoint for autoresearch-spark sandbox.
#
# Starts (or reattaches to) a zellij session with claude running inside.
# When claude exits, the pane drops to a shell so you can inspect and re-launch.
# New tabs/panes use the default shell (bash).
#
# Usage:
#   openshell sandbox create --from Dockerfile --gpu -- sandbox/start-with-zellij.sh

SESSION="autoresearch"

# Initialize git repo if not already set up (program.md expects git for branching)
if [ ! -d /sandbox/autoresearch/.git ]; then
    git config --global user.email "sandbox@autoresearch"
    git config --global user.name "autoresearch"
    git -C /sandbox/autoresearch init -q
    git -C /sandbox/autoresearch add -A
    git -C /sandbox/autoresearch commit -q -m "initial commit"
fi

# Write zellij config and layout.
mkdir -p ~/.config/zellij
cat > ~/.config/zellij/config.kdl << 'CFG'
show_release_notes false
show_startup_tips false
default_mode "locked"
CFG

cat > /tmp/autoresearch-layout.kdl << 'KDL'
layout {
    pane size=1 borderless=true {
        plugin location="tab-bar"
    }
    pane size="75%" {
        command "bash"
        args "-c" "claude --dangerously-skip-permissions; exec bash"
        cwd "/sandbox/autoresearch"
    }
    pane size="25%" {
        command "tail"
        args "-F" "run.log"
        cwd "/sandbox/autoresearch"
    }
    pane size=1 borderless=true {
        plugin location="status-bar"
    }
}
KDL

# Attach to existing session, or create a new one with the layout.
SESSIONS=$(zellij list-sessions --no-formatting 2>/dev/null) || true

if echo "$SESSIONS" | grep -q "^$SESSION.*EXITED"; then
    zellij delete-session "$SESSION" 2>/dev/null || true
    exec zellij -s "$SESSION" -n /tmp/autoresearch-layout.kdl
elif echo "$SESSIONS" | grep -q "^$SESSION"; then
    exec zellij attach "$SESSION"
else
    exec zellij -s "$SESSION" -n /tmp/autoresearch-layout.kdl
fi
