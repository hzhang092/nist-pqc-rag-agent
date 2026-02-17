# Mini Retrieval Sanity (Base vs Hybrid+Fusion)

## ML-KEM.KeyGen
- base spec-like@5: 5 | hybrid spec-like@5: 5 | delta: 0
- base top-5:
  - NIST.FIPS.203 p44-p44 (NIST.FIPS.203::p0044::c001) spec_like=True
  - NIST.FIPS.203 p25-p25 (NIST.FIPS.203::p0025::c000) spec_like=True
  - NIST.FIPS.203 p26-p26 (NIST.FIPS.203::p0026::c000) spec_like=True
  - NIST.FIPS.203 p45-p45 (NIST.FIPS.203::p0045::c001) spec_like=True
  - NIST.FIPS.203 p41-p41 (NIST.FIPS.203::p0041::c001) spec_like=True
- hybrid top-5:
  - NIST.FIPS.203 p44-p44 (NIST.FIPS.203::p0044::c001) spec_like=True
  - NIST.FIPS.203 p26-p26 (NIST.FIPS.203::p0026::c000) spec_like=True
  - NIST.FIPS.203 p25-p25 (NIST.FIPS.203::p0025::c000) spec_like=True
  - NIST.FIPS.203 p41-p41 (NIST.FIPS.203::p0041::c001) spec_like=True
  - NIST.FIPS.203 p45-p45 (NIST.FIPS.203::p0045::c001) spec_like=True

## Algorithm 19 ML-KEM.KeyGen
- base spec-like@5: 4 | hybrid spec-like@5: 5 | delta: 1
- base top-5:
  - NIST.FIPS.203 p44-p44 (NIST.FIPS.203::p0044::c002) spec_like=True
  - NIST.FIPS.203 p26-p26 (NIST.FIPS.203::p0026::c000) spec_like=True
  - NIST.FIPS.203 p45-p45 (NIST.FIPS.203::p0045::c001) spec_like=True
  - NIST.FIPS.204 p9-p9 (NIST.FIPS.204::p0009::c002) spec_like=False
  - NIST.FIPS.203 p41-p41 (NIST.FIPS.203::p0041::c001) spec_like=True
- hybrid top-5:
  - NIST.FIPS.203 p26-p26 (NIST.FIPS.203::p0026::c000) spec_like=True
  - NIST.FIPS.203 p9-p9 (NIST.FIPS.203::p0009::c001) spec_like=True
  - NIST.FIPS.203 p25-p25 (NIST.FIPS.203::p0025::c000) spec_like=True
  - NIST.FIPS.203 p23-p23 (NIST.FIPS.203::p0023::c002) spec_like=True
  - NIST.FIPS.203 p41-p41 (NIST.FIPS.203::p0041::c001) spec_like=True

## K-PKE Key Generation
- base spec-like@5: 3 | hybrid spec-like@5: 5 | delta: 2
- base top-5:
  - NIST.FIPS.203 p8-p8 (NIST.FIPS.203::p0008::c000) spec_like=True
  - NIST.FIPS.203 p9-p9 (NIST.FIPS.203::p0009::c001) spec_like=True
  - NIST.SP.800-227 p8-p8 (NIST.SP.800-227::p0008::c001) spec_like=False
  - NIST.FIPS.203 p37-p37 (NIST.FIPS.203::p0037::c001) spec_like=True
  - NIST.FIPS.205 p8-p8 (NIST.FIPS.205::p0008::c000) spec_like=False
- hybrid top-5:
  - NIST.FIPS.203 p8-p8 (NIST.FIPS.203::p0008::c000) spec_like=True
  - NIST.FIPS.203 p37-p37 (NIST.FIPS.203::p0037::c001) spec_like=True
  - NIST.FIPS.203 p41-p41 (NIST.FIPS.203::p0041::c001) spec_like=True
  - NIST.FIPS.203 p48-p48 (NIST.FIPS.203::p0048::c000) spec_like=True
  - NIST.FIPS.203 p9-p9 (NIST.FIPS.203::p0009::c001) spec_like=True

## ML-KEM-768 parameter set
- base spec-like@5: 3 | hybrid spec-like@5: 4 | delta: 1
- base top-5:
  - NIST.FIPS.203 p48-p48 (NIST.FIPS.203::p0048::c000) spec_like=True
  - NIST.FIPS.203 p23-p23 (NIST.FIPS.203::p0023::c002) spec_like=True
  - NIST.FIPS.204 p9-p9 (NIST.FIPS.204::p0009::c001) spec_like=False
  - NIST.FIPS.204 p25-p25 (NIST.FIPS.204::p0025::c000) spec_like=False
  - NIST.FIPS.203 p44-p44 (NIST.FIPS.203::p0044::c001) spec_like=True
- hybrid top-5:
  - NIST.FIPS.203 p49-p49 (NIST.FIPS.203::p0049::c001) spec_like=True
  - NIST.FIPS.203 p23-p23 (NIST.FIPS.203::p0023::c002) spec_like=True
  - NIST.SP.800-227 p43-p43 (NIST.SP.800-227::p0043::c001) spec_like=False
  - NIST.FIPS.203 p25-p25 (NIST.FIPS.203::p0025::c000) spec_like=True
  - NIST.FIPS.203 p48-p48 (NIST.FIPS.203::p0048::c001) spec_like=True

## decapsulation algorithm ML-KEM.Decaps
- base spec-like@5: 5 | hybrid spec-like@5: 5 | delta: 0
- base top-5:
  - NIST.FIPS.203 p46-p46 (NIST.FIPS.203::p0046::c001) spec_like=True
  - NIST.FIPS.203 p44-p44 (NIST.FIPS.203::p0044::c000) spec_like=True
  - NIST.FIPS.203 p42-p42 (NIST.FIPS.203::p0042::c001) spec_like=True
  - NIST.FIPS.203 p47-p47 (NIST.FIPS.203::p0047::c000) spec_like=True
  - NIST.FIPS.203 p23-p23 (NIST.FIPS.203::p0023::c002) spec_like=True
- hybrid top-5:
  - NIST.FIPS.203 p46-p46 (NIST.FIPS.203::p0046::c001) spec_like=True
  - NIST.FIPS.203 p44-p44 (NIST.FIPS.203::p0044::c000) spec_like=True
  - NIST.FIPS.203 p23-p23 (NIST.FIPS.203::p0023::c002) spec_like=True
  - NIST.FIPS.203 p47-p47 (NIST.FIPS.203::p0047::c000) spec_like=True
  - NIST.FIPS.203 p26-p26 (NIST.FIPS.203::p0026::c000) spec_like=True
