#!/usr/bin/env bash

set -euo pipefail

readonly program_name=${0##*/}

usage() {
    cat <<EOF
Usage:
  sudo $program_name [--user USER]
  sudo $program_name --remove [--user USER]

Install or remove a persistent sudoers rule granting USER passwordless sudo.
When --user is omitted, the original user from SUDO_USER is used.
EOF
}

fatal() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

action=install
target_user=${SUDO_USER:-}

while (( $# > 0 )); do
    case $1 in
        --remove)
            action=remove
            shift
            ;;
        --user)
            (( $# >= 2 )) || fatal "--user requires a value"
            target_user=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fatal "unknown argument: $1"
            ;;
    esac
done

(( EUID == 0 )) || fatal "run this script with sudo"
[[ -n $target_user ]] || fatal "cannot determine the target user; pass --user USER"
[[ $target_user != root ]] || fatal "refusing to create a redundant rule for root"
[[ $target_user =~ ^[A-Za-z_][A-Za-z0-9_.-]*$ ]] || fatal "unsafe user name: $target_user"
id "$target_user" >/dev/null 2>&1 || fatal "user does not exist: $target_user"

visudo_bin=$(command -v visudo) || fatal "visudo is not installed"
readonly visudo_bin
readonly rule_path="/etc/sudoers.d/90-${target_user}-nopasswd"
readonly rule_text="${target_user} ALL=(ALL:ALL) NOPASSWD: ALL"

if [[ $action == remove ]]; then
    if [[ ! -e $rule_path ]]; then
        printf 'No rule to remove: %s\n' "$rule_path"
        exit 0
    fi
    rm -f -- "$rule_path"
    "$visudo_bin" -cf /etc/sudoers
    printf 'Removed %s; passwordless sudo is disabled for %s.\n' "$rule_path" "$target_user"
    exit 0
fi

if [[ -e $rule_path ]]; then
    existing_rule=$(tr -d '\r\n' < "$rule_path")
    [[ $existing_rule == "$rule_text" ]] || fatal "$rule_path already exists with different contents"
    chown root:root "$rule_path"
    chmod 0440 "$rule_path"
    "$visudo_bin" -cf /etc/sudoers
    printf 'Rule already installed and valid: %s\n' "$rule_path"
    exit 0
fi

temp_rule=$(mktemp "/etc/sudoers.d/.${program_name}.XXXXXX")
readonly temp_rule
cleanup() {
    rm -f -- "$temp_rule"
}
trap cleanup EXIT

printf '%s\n' "$rule_text" > "$temp_rule"
chown root:root "$temp_rule"
chmod 0440 "$temp_rule"
"$visudo_bin" -cf "$temp_rule"
mv -T -- "$temp_rule" "$rule_path"
trap - EXIT

"$visudo_bin" -cf /etc/sudoers
printf 'Installed persistent passwordless sudo for %s: %s\n' "$target_user" "$rule_path"
printf 'Warning: this grants %s unrestricted root access without a password.\n' "$target_user"
printf 'To undo: sudo %s --remove --user %s\n' "$0" "$target_user"
