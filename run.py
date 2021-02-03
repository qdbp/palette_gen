import pickle

from palette_gen.palette_solver import (
    ColorScheme, N_BRIGHT, N_CFUL, dump_color_scheme,
    solve_uniform_color_grid, )

Exuberance = ColorScheme(
    name="Exuberance",
    bg_hex="#161616",
    # ultras
    ultra_brg=(11, N_BRIGHT),
    ultras_sat=(14, N_CFUL),
    # primaries
    prim_brg=(11, 14),
    prim_sat=(10, 13),
    # secondaries
    sec_brg=(5, 7),
    sec_sat=(1, 4),
    # highlights
    hl_brg=(6, 7),
    hl_sat=(7, 11),
    # backgrounds
    bg_brg=(2, 4),
    bg_sat=(2, 7)
)

Restraint = ColorScheme(
    name="Restraint",
    bg_hex="#d9d9d9",
    # ultras
    ultra_brg=(9, 11),
    ultras_sat=(13, N_CFUL),
    # primares
    prim_brg=(6, 8),
    prim_sat=(6, N_CFUL),
    # secondaries
    sec_brg=(9, 12),
    sec_sat=(1, 3),
    # highlights
    hl_brg=(11, 14),
    hl_sat=(4, 9),
    # backgrounds
    bg_brg=(12, N_BRIGHT),
    bg_sat=(1, 4)
)

Split = ColorScheme(
    name="Split",
    bg_hex="#808080",
    # ultras
    ultra_brg=(11, 13),
    ultras_sat=(13, N_CFUL),
    # primares
    prim_brg=(3, 12),
    prim_sat=(5, 12),
    # secondaries
    sec_brg=(5, 12),
    sec_sat=(1, 4),
    # highlights
    hl_brg=(13, 14),
    hl_sat=(6, 8),
    # backgrounds
    bg_brg=(8, 10),
    bg_sat=(2, 6),
)


if __name__ == "__main__":
    # solve_uniform_color_grid('base_colors.p', do_plot=False)
    with open("base_colors.p", "rb") as f:
        all_colors = pickle.load(f)

    # dump_color_scheme(all_colors, bg_rgb='#161616')
    # Exuberance.generate(all_colors)

    dump_color_scheme(all_colors, bg_rgb='#e0e0e0')
    # Restraint.generate(all_colors)

    # dump_color_scheme(all_colors, bg_rgb='#808080')
    # Split.generate(all_colors)

