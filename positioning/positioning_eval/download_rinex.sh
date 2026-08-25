#!/bin/bash

###############################################################################
##  RINEX Download Script for CDDIS Archive                                 ##
##  Downloads observation files for specified stations and date             ##
###############################################################################

# --- Retry/timeout schedule --------------------------------------------------
#
# A real successful download of a 6.6 MB file takes ~5s against CDDIS, so these
# bounds exist only to give up on a genuinely stuck connection - they do not need to
# be generous to the happy path. This schedule is deliberately smaller than the one
# it replaced (5 attempts x 5/10/20/40/80s backoff, wget -t 3, and no timeout at all
# on the directory-listing fetch below), which let a single station-day run past
# 300s and taught download_rinex.py's old 120s subprocess timeout to fire mid-retry
# on nearly every transient CDDIS hiccup - confirmed as the actual cause of every
# "no RINEX" skip in the 2026-08 station-recovery sweep (1,491/1,491 skips preceded
# by a timeout, all firing at exactly 120-121s), not missing data: a stratified
# re-check found the file present on CDDIS 15/15 times.
#
# Worst case for ONE wget invocation, all WGET_TRIES exhausted:
#   WGET_TRIES * (WGET_CONNECT_TIMEOUT_S + WGET_READ_TIMEOUT_S) = 2 * (10 + 60) = 140s
#
# Worst case for one filename format's retry loop (RETRY_MAX_ATTEMPTS wget
# invocations, backoff sleep between all but the last attempt):
#   RETRY_MAX_ATTEMPTS * 140 + (RETRY_INITIAL_DELAY_S + RETRY_INITIAL_DELAY_S*2)
#   = 3 * 140 + (5 + 10) = 420 + 15 = 435s
#
# download_rinex() tries the RINEX3 long format, then (if that also fails) the
# RINEX2 short format - each with its own full retry loop above - plus one
# get_directory_listing() call (same wget bounds, single attempt) when the
# per-batch cache is cold:
#   2 * 435 + 140 = 1010s
#
# download_rinex.py's RINEX_DOWNLOAD_TIMEOUT_SECONDS must stay above this 1010s
# figure, with headroom - see that constant's own comment.
# tests/positioning/test_download_rinex_timeout.py parses these constants back out
# of this file and re-derives 1010s independently, so a future change to the retry
# schedule that pushes the real worst case above the Python timeout fails that test
# instead of silently reintroducing the bug.
readonly WGET_CONNECT_TIMEOUT_S=10
readonly WGET_READ_TIMEOUT_S=60
readonly WGET_TRIES=2
readonly RETRY_MAX_ATTEMPTS=3
readonly RETRY_INITIAL_DELAY_S=5

# Cache directory listing to avoid repeated requests
if [ -n "$RINEX_CACHE_DIR" ]; then
    CACHE_DIR="$RINEX_CACHE_DIR"
else
    CACHE_DIR="/tmp/rinex_cache_$$"
    trap 'rm -rf "$CACHE_DIR"' EXIT
fi
mkdir -p "$CACHE_DIR"

get_directory_listing() {
    local year="$1"
    local doy="$2"
    local yy="${year:2:2}"
    local base_url="https://cddis.nasa.gov/archive/gnss/data/daily/${year}/${doy}/${yy}d"
    local cache_file="$CACHE_DIR/listing_${year}_${doy}.txt"

    # Return cached listing if it exists. download_rinex_batch() runs many stations
    # concurrently against a shared $CACHE_DIR, so multiple threads race here for the
    # same year/doy whenever a day has more than one station.
    if [ -f "$cache_file" ]; then
        cat "$cache_file"
        return 0
    fi

    # Fetch into a unique temp file on the same filesystem as $cache_file (mktemp with
    # the cache file's own path as a template guarantees that), then rename into place.
    # `mv` within one filesystem is atomic, so a concurrent reader of $cache_file never
    # observes a half-written listing - it either doesn't exist yet or is the complete
    # file. A second thread racing this fetch just re-downloads the same listing and
    # overwrites it with equivalent content; wasteful but harmless, and simpler than
    # locking out concurrent fetches entirely - only this listing fetch needs
    # protecting, not the whole per-station download.
    #
    # The previous version piped `wget -O - | tee "$cache_file"`: two concurrent tee
    # processes writing the same path can interleave, and on a failed wget it still
    # left a cache file behind (empty or truncated), which every later call - not just
    # the racing ones - would then treat as a valid cached listing forever.
    local tmp_file
    tmp_file="$(mktemp "${cache_file}.XXXXXX")"
    if wget --netrc --auth-no-challenge -t "$WGET_TRIES" \
            --connect-timeout="$WGET_CONNECT_TIMEOUT_S" \
            --read-timeout="$WGET_READ_TIMEOUT_S" \
            -q -O "$tmp_file" "${base_url}/" 2>/dev/null; then
        mv -f "$tmp_file" "$cache_file"
        cat "$cache_file"
    else
        rm -f "$tmp_file"
        # No cache file written on failure, so the next caller retries the fetch
        # instead of inheriting an empty/failed listing.
    fi
}

download_rinex() {
    # Usage: download_rinex STATION YEAR DOY OUTPUT_DIR [KNOWN_FILENAME]
    local station="$1"
    local year="$2"
    local doy="$3"
    local output_dir="$4"
    local known_filename="$5"  # Optional: if provided, use this exact filename

    local station_upper="${station^^}"
    local station_lower="${station,,}"
    local yy="${year:2:2}"

    mkdir -p "$output_dir"

    # Try RINEX 3 long filename format - use wildcard to match any country code
    # Format: SSSS00CCC_R_YYYYDDDHHMM_01D_30S_MO.crx.gz
    # where SSSS=station, CCC=3-letter country code
    local long_pattern="${station_upper}00???_R_${year}${doy}0000_01D_30S_MO.crx.gz"
    local base_url="https://cddis.nasa.gov/archive/gnss/data/daily/${year}/${doy}/${yy}d"

    # Try RINEX 2 short filename format
    local short_filename="${station_lower}${doy}0.${yy}d.Z"
    local short_url="https://cddis.nasa.gov/archive/gnss/data/daily/${year}/${doy}/${yy}d/${short_filename}"

    cd "$output_dir" || return 1

    # Create empty cookies file if it doesn't exist
    touch cookies.txt

    # Get directory listing (cached) and find the matching file for this station
    local dir_listing=$(get_directory_listing "$year" "$doy")

    # Extract the RINEX 3 filename for this station (any country code)
    local long_filename=$(echo "$dir_listing" | grep -oE "${station_upper}00[A-Z]{3}_R_${year}${doy}0000_01D_30S_MO\.crx\.gz" | head -1)

    if [ -n "$long_filename" ]; then
        # Found the file - download it
        local long_url="${base_url}/${long_filename}"

        # Download with authentication and retries
        local attempt=1
        local delay=$RETRY_INITIAL_DELAY_S
        local success=0

        while [ $attempt -le $RETRY_MAX_ATTEMPTS ]; do
            # Clean up any previous failed attempts
            [ -f "$long_filename" ] && ! file "$long_filename" | grep -q "gzip compressed" && rm -f "$long_filename"

            if wget --netrc --auth-no-challenge=on --keep-session-cookies --save-cookies=cookies.txt --load-cookies=cookies.txt -nv -nc -c \
                    -t "$WGET_TRIES" --connect-timeout="$WGET_CONNECT_TIMEOUT_S" --read-timeout="$WGET_READ_TIMEOUT_S" "$long_url" 2>&1; then
                if [ -f "$long_filename" ] && file "$long_filename" | grep -q "gzip compressed"; then
                    success=1
                    break
                else
                    [ -f "$long_filename" ] && rm -f "$long_filename"
                fi
            fi

            if [ $attempt -lt $RETRY_MAX_ATTEMPTS ]; then
                echo "Attempt $attempt of $RETRY_MAX_ATTEMPTS failed for $long_filename. Retrying in ${delay}s..."
                sleep $delay
                delay=$((delay * 2))
            fi
            attempt=$((attempt + 1))
        done

        if [ $success -eq 1 ]; then
            gunzip -f "$long_filename" 2>/dev/null
            local crx_file="${long_filename%.gz}"
            if [ -f "$crx_file" ]; then
                # Convert Hatanaka to RINEX
                local converter=""
                if command -v CRX2RNX &> /dev/null; then
                    converter="CRX2RNX"
                elif command -v crx2rnx &> /dev/null; then
                    converter="crx2rnx"
                elif [ -f "$HOME/.local/bin/crx2rnx" ]; then
                    converter="$HOME/.local/bin/crx2rnx"
                fi

                if [ -n "$converter" ]; then
                    $converter "$crx_file" 2>/dev/null
                    rm -f "$crx_file"
                    echo "Downloaded and converted: ${station_upper}"
                    return 0
                else
                    echo "Downloaded: ${station_upper} (Hatanaka .crx format)"
                    return 0
                fi
            fi
        fi
    fi

    # Try short format (use ~/.netrc for authentication with NASA Earthdata)
    local attempt=1
    local delay=$RETRY_INITIAL_DELAY_S
    local success=0

    while [ $attempt -le $RETRY_MAX_ATTEMPTS ]; do
        # Clean up any previous failed short format downloads
        [ -f "$short_filename" ] && ! file "$short_filename" | grep -q "compress'd data" && rm -f "$short_filename"

        if wget --netrc --auth-no-challenge=on --keep-session-cookies --load-cookies=cookies.txt -nv -nc -c \
                -t "$WGET_TRIES" --connect-timeout="$WGET_CONNECT_TIMEOUT_S" --read-timeout="$WGET_READ_TIMEOUT_S" "$short_url" 2>&1; then
            if [ -f "$short_filename" ] && file "$short_filename" | grep -q "compress'd data"; then
                success=1
                break
            else
                rm -f "$short_filename"
            fi
        fi

        if [ $attempt -lt $RETRY_MAX_ATTEMPTS ]; then
            echo "Attempt $attempt of $RETRY_MAX_ATTEMPTS failed for $short_filename. Retrying in ${delay}s..."
            sleep $delay
            delay=$((delay * 2))
        fi
        attempt=$((attempt + 1))
    done

    if [ $success -eq 1 ]; then
        uncompress "$short_filename" 2>/dev/null
        echo "Downloaded: ${station_upper}"
        return 0
    fi

    echo "Failed to download RINEX for ${station_upper}"
    return 1
}

# Main entry point - only when executed directly, not when sourced. Tests source this
# file (to exercise get_directory_listing()/download_rinex() with a fake wget on PATH,
# with no network access) and rely on sourcing not triggering a real download.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ $# -ge 3 ]; then
        download_rinex "$1" "$2" "$3" "${4:-.}"
    else
        echo "Usage: download_rinex.sh STATION YEAR DOY [OUTPUT_DIR]"
        echo "Example: download_rinex.sh ZIMM 2024 183 ./rinex"
        exit 1
    fi
fi
