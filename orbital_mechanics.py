
import numpy as np
import datetime

def checksum(line):
    """Compute TLE checksum."""
    s = 0
    for c in line[:-1]:
        if c.isdigit():
            s += int(c)
        elif c == '-':
            s += 1
    return s % 10

def generate_walker_delta(n_sats=24, n_planes=3, inclination=53.0, altitude_km=550.0):
    """
    Generate TLEs for a Walker Delta constellation.
    Format: i:T/P/F where i=inclination, T=total sats, P=planes, F=phasing
    We assume symmetric distribution.
    Returns a list of (line1, line2) tuples.
    """
    sats_per_plane = n_sats // n_planes
    
    # Constants
    mu = 3.986004418e14  # m^3/s^2
    Re = 6378.137  # km
    semimajor_axis = Re + altitude_km
    # Mean motion in revs per day
    n_rad_s = np.sqrt(mu / ((semimajor_axis * 1000)**3))
    n_rev_day = n_rad_s * (24 * 3600) / (2 * np.pi)
    
    tles = []
    
    sat_id = 10001
    epoch_year = 26  # 2026
    epoch_day = 1.0  # Jan 1st
    
    for p in range(n_planes):
        raan = (360.0 / n_planes) * p
        for s in range(sats_per_plane):
            # True anomaly spacing
            mean_anomaly = (360.0 / sats_per_plane) * s
            
            # Phasing between planes (F=1 for Walker Delta usually implies offset)
            # Simple version: offset anomaly by p * (360 / T)
            phasing_offset = p * (360.0 / n_sats) 
            final_anomaly = (mean_anomaly + phasing_offset) % 360.0
            
            # Construct TLE Line 1
            # 1 NNNNNC 26001A   26001.00000000  .00000000  00000-0  00000-0 0  999X
            # We'll compute checksum at end
            line1_template = "1 {0:05d}U 26001A   {1:02d}{2:012.8f}  .00000000  00000-0  00000-0 0  999"
            line1_raw = line1_template.format(sat_id, epoch_year, epoch_day)
            line1 = line1_raw + str(checksum(line1_raw + "0"))
            
            # Construct TLE Line 2
            # 2 NNNNN IIII.IIII RRRR.RRRR EEEEEEE AAAA.AAAA MMMM.MMMM NN.NNNNNNNNRRRRX
            # i, raan, ecc, arg_p, ma, mean_motion, rev_num
            eccentricity = 0.0001 # Circular
            arg_perigee = 0.0
            
            line2_template = "2 {0:05d} {1:8.4f} {2:8.4f} {3:07d} {4:8.4f} {5:8.4f} {6:11.8f}    0"
            # Format requires leading spaces sometimes
            line2_raw = line2_template.format(
                sat_id,
                inclination,
                raan,
                int(eccentricity * 1e7),
                arg_perigee,
                final_anomaly,
                n_rev_day
            )
            line2 = line2_raw + str(checksum(line2_raw + "0"))
            
            tles.append((line1, line2))
            sat_id += 1
            
    return tles

if __name__ == "__main__":
    tles = generate_walker_delta()
    print(f"Generated {len(tles)} satellites")
    print("Sample TLE:")
    print(tles[0][0])
    print(tles[0][1])
