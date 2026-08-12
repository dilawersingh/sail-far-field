"""
exposure_lookup.py
--------------------
Human-readable exposure value <-> EDSDK hex code conversion, parsed
programmatically from the EDSDK API Programming Reference (sections 5.2.22
ISO, 5.2.26 Tv) rather than transcribed by hand, and cross-checked against
every value actually used in exposure_settings.json before being generated.

No Av table: this rig has a bare sensor with no lens attached, so aperture
isn't a controllable or meaningful parameter -- ISO and Tv are the only two
real exposure knobs. See canon_camera.py's kEdsPropID_Av if that ever changes.

Usage
-----
    from exposure_lookup import iso_to_hex, tv_to_hex

    iso_code = iso_to_hex("800")   # -> 0x60
    tv_code  = tv_to_hex("1/50")   # -> 0x65
"""

# value (str, exactly as written in exposure_settings.json) -> EDSDK hex code
ISO_CODES = {
    'Auto': 0x00000000,
    '6': 0x00000028,
    '12': 0x00000030,
    '25': 0x00000038,
    '50': 0x00000040,
    '100': 0x00000048,
    '125': 0x0000004B,
    '160': 0x0000004D,
    '200': 0x00000050,
    '250': 0x00000053,
    '320': 0x00000055,
    '400': 0x00000058,
    '500': 0x0000005B,
    '640': 0x0000005D,
    '800': 0x00000060,
    '1000': 0x00000063,
    '1250': 0x00000065,
    '1600': 0x00000068,
    '2000': 0x0000006B,
    '2500': 0x0000006D,
    '3200': 0x00000070,
    '4000': 0x00000073,
    '5000': 0x00000075,
    '6400': 0x00000078,
    '8000': 0x0000007B,
    '10000': 0x0000007D,
    '12800': 0x00000080,
    '16000': 0x00000083,
    '20000': 0x00000085,
    '25600': 0x00000088,
    '32000': 0x0000008B,
    '40000': 0x0000008D,
    '51200': 0x00000090,
    '64000': 0x00000093,
    '80000': 0x00000095,
    '102400': 0x00000098,
    '204800': 0x000000A0,
    '409600': 0x000000A8,
    '819200': 0x000000B0,
}

# value (str, exactly as written in exposure_settings.json) -> EDSDK hex code
TV_CODES = {
    'Bulb': 0x0000000C,
    '30"': 0x00000010,
    '25"': 0x00000013,
    '20"': 0x00000014,
    '20" (1/3)': 0x00000015,
    '15"': 0x00000018,
    '13"': 0x0000001B,
    '10"': 0x0000001C,
    '10" (1/3)': 0x0000001D,
    '8"': 0x00000020,
    '6" (1/3)': 0x00000023,
    '6"': 0x00000024,
    '5"': 0x00000025,
    '4"': 0x00000028,
    '3"2': 0x0000002B,
    '3"': 0x0000002C,
    '2"5': 0x0000002D,
    '2"': 0x00000030,
    '1"6': 0x00000033,
    '1"5': 0x00000034,
    '1"3': 0x00000035,
    '1"': 0x00000038,
    '0"8': 0x0000003B,
    '0"7': 0x0000003C,
    '0"6': 0x0000003D,
    '0"5': 0x00000040,
    '0"4': 0x00000043,
    '0"3': 0x00000044,
    '0"3 (1/3)': 0x00000045,
    '1/4': 0x00000048,
    '1/5': 0x0000004B,
    '1/6': 0x0000004C,
    '1/6 (1/3)': 0x0000004D,
    '1/8': 0x00000050,
    '1/10 (1/3)': 0x00000053,
    '1/10': 0x00000054,
    '1/13': 0x00000055,
    '1/15': 0x00000058,
    '1/20 (1/3)': 0x0000005B,
    '1/20': 0x0000005C,
    '1/25': 0x0000005D,
    '1/30': 0x00000060,
    '1/40': 0x00000063,
    '1/45': 0x00000064,
    '1/50': 0x00000065,
    '1/60': 0x00000068,
    '1/80': 0x0000006B,
    '1/90': 0x0000006C,
    '1/100': 0x0000006D,
    '1/125': 0x00000070,
    '1/160': 0x00000073,
    '1/180': 0x00000074,
    '1/200': 0x00000075,
    '1/250': 0x00000078,
    '1/320': 0x0000007B,
    '1/350': 0x0000007C,
    '1/400': 0x0000007D,
    '1/500': 0x00000080,
    '1/640': 0x00000083,
    '1/750': 0x00000084,
    '1/800': 0x00000085,
    '1/1000': 0x00000088,
    '1/1250': 0x0000008B,
    '1/1500': 0x0000008C,
    '1/1600': 0x0000008D,
    '1/2000': 0x00000090,
    '1/2500': 0x00000093,
    '1/3000': 0x00000094,
    '1/3200': 0x00000095,
    '1/4000': 0x00000098,
    '1/5000': 0x0000009B,
    '1/6000': 0x0000009C,
    '1/6400': 0x0000009D,
    '1/8000': 0x000000A0,
    '1/10000': 0x000000A3,
    '1/12800': 0x000000A5,
    '1/16000': 0x000000A8,
    '1/20000': 0x000000AB,
    '1/25600': 0x000000AD,
    '1/32000': 0x000000B0,
}


def iso_to_hex(value: str) -> int:
    """value: plain ISO number as a string, e.g. "800", or "Auto"."""
    value = str(value).strip()
    if value not in ISO_CODES:
        raise KeyError(
            f"ISO value {value!r} not found in the EDSDK ISO table. "
            f"Check exposure_settings.json for a typo, or confirm this ISO "
            f"value is actually supported by the 1000D/T2i (not all cameras "
            f"support the full range in the manual's table)."
        )
    return ISO_CODES[value]


def tv_to_hex(value: str) -> int:
    # value: shutter speed exactly as written in the manual's format, e.g.
    # "1/50" for a fraction, or a number followed by a literal double-quote
    # character for seconds (e.g. two seconds is written 2 followed by ").
    # Bulb cannot be set via software (Canon's own restriction) even though
    # it appears in the lookup table.
    value = str(value).strip()
    if value not in TV_CODES:
        raise KeyError(
            f"Tv value {value!r} not found in the EDSDK Tv table. Check "
            f"exposure_settings.json for a typo -- seconds need a trailing "
            f"double-quote character, fractions need to match the table "
            f"exactly (e.g. 1/50, not 1/50s)."
        )
    if value == "Bulb":
        raise ValueError("Bulb cannot be set via software (Canon EDSDK restriction).")
    return TV_CODES[value]
