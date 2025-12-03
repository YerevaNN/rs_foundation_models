def normalize_band_names(bands):
    """
    Normalize band names from short format (B2, B3, B4) to long format (B02, B03, B04).
    
    Args:
        bands: List of band names like ['B4', 'B3', 'B2'] or single string 'B2'
    
    Returns:
        Normalized band names like ['B04', 'B03', 'B02'] or 'B02'
    """
    import re
    
    def normalize_single(band):
        # Match B followed by single digit (B1-B9), but not B01, B11, B12, B8A, VV, VH
        if re.match(r'^B([1-9])$', band):
            number = re.match(r'^B([1-9])$', band).group(1)
            return f'B{number.zfill(2)}'  # B2 -> B02, B3 -> B03, B4 -> B04
        return band  # Return as-is for B02, B11, B12, B8A, VV, VH, etc.
    
    if isinstance(bands, str):
        return normalize_single(bands)
    elif isinstance(bands, list):
        return [normalize_single(band) for band in bands]
    return bands


# Example usage:
if __name__ == '__main__':
    # Test with the example
    input_bands = ['B4', 'B3', 'B2']
    normalized = normalize_band_names(input_bands)
    print(f"Input:  {input_bands}")
    print(f"Output: {normalized}")
    # Output: ['B04', 'B03', 'B02']

