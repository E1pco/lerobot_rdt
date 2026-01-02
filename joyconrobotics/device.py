from __future__ import annotations

from typing import List, Optional, Tuple

import hid

from .constants import JOYCON_L_PRODUCT_ID, JOYCON_PRODUCT_IDS, JOYCON_R_PRODUCT_ID, JOYCON_VENDOR_ID


def get_device_ids(debug: bool = False) -> List[Tuple[int, int, Optional[str]]]:
    """Return Joy-Con HID ids as a list of `(vendor_id, product_id, serial_number)`.

    Notes:
    - Always returns a list (possibly empty).
    - `serial_number` can be `None` depending on HID backend / permissions.
    """
    devices = hid.enumerate(0, 0)

    out: List[Tuple[int, int, Optional[str]]] = []
    for dev in devices:
        vendor_id = int(dev.get("vendor_id", 0) or 0)
        product_id = int(dev.get("product_id", 0) or 0)

        if vendor_id != JOYCON_VENDOR_ID:
            continue
        if product_id not in JOYCON_PRODUCT_IDS:
            continue

        product_string = dev.get("product_string")
        serial = dev.get("serial") or dev.get("serial_number")

        out.append((vendor_id, product_id, serial))

        if debug:
            print(product_string or "")
            print(f"\tvendor_id  is {vendor_id!r}")
            print(f"\tproduct_id is {product_id!r}")
            print(f"\tserial     is {serial!r}")

    return out


def is_id_L(id):
    return id[1] == JOYCON_L_PRODUCT_ID


def get_ids_of_type(lr, **kw):
    """
    returns a list of tuples like `(vendor_id, product_id, serial_number)`

    arg: lr : str : put `R` or `L`
    """
    if lr.lower() == "l":
        product_id = JOYCON_L_PRODUCT_ID
    else:
        product_id = JOYCON_R_PRODUCT_ID
    return [i for i in get_device_ids(**kw) if i[1] == product_id]


def get_R_ids(**kw):
    """returns a list of tuple like `(vendor_id, product_id, serial_number)`"""
    return get_ids_of_type("R", **kw)


def get_L_ids(**kw):
    """returns a list of tuple like `(vendor_id, product_id, serial_number)`"""
    return get_ids_of_type("L", **kw)


def get_R_id(**kw):
    """returns a tuple like `(vendor_id, product_id, serial_number)`"""
    ids = get_R_ids(**kw)
    if not ids:
        return (None, None, None)
    return ids[0]


def get_L_id(**kw):
    """returns a tuple like `(vendor_id, product_id, serial_number)`"""
    ids = get_L_ids(**kw)
    if not ids:
        return (None, None, None)
    return ids[0]


if __name__ == "__main__":
    def _print_ids(title: str, ids: List[Tuple[int, int, Optional[str]]]) -> None:
        print(title)
        if not ids:
            print("  (none)")
            return
        for vendor_id, product_id, serial in ids:
            lr = "L" if product_id == JOYCON_L_PRODUCT_ID else ("R" if product_id == JOYCON_R_PRODUCT_ID else "?")
            print(f"  Joy-Con ({lr}) vendor_id={vendor_id} product_id={product_id} serial={serial!r}")

    _print_ids("All Joy-Con IDs:", get_device_ids(debug=False))
    print()
    _print_ids("Left Joy-Con IDs:", get_L_ids(debug=False))
    print()
    _print_ids("Right Joy-Con IDs:", get_R_ids(debug=False))