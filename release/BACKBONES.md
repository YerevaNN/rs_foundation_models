# Evaluated backbone manifest

This covers all 14 evaluated backbones. [backbones.json](backbones.json) is normative. Only χViT currently has byte-level public-to-evaluated parity. Every entry gives the exact loader and, where identifiable, an immutable public candidate. Candidate filename agreement alone does not prove that the evaluated bytes were identical.

| Model | Status | Candidate revision | Remaining check |
|---|---|---|---|
| ResNet50 | unresolved | — | Recover the actual timm version and cached weight SHA-256; the declared timm version is incompatible with the checked-in encoder API. |
| ViT-B | unresolved | — | Recover the actual timm version and cached weight SHA-256. |
| iBOT | unresolved | — | Executed configs select the internal MillionAID checkpoint, while the manuscript says ImageNet. Identify, publish, checksum, and license the evaluated checkpoint. |
| DINOv2 | unresolved | facebook/dinov2-base@f9e44c814b77 | The evaluator used an unpinned torch.hub branch. Hash the cluster cache and identify the matching source revision. |
| χViT | verified | yerevann/ChiViT@dad320e530f4 | Complete |
| DOFA | filename_match_unverified | earthflow/DOFA@7a5219e48d2f | Hash the cluster file to prove parity. |
| CROMA | filename_match_unverified | antofuller/CROMA@0dd28e3d633b | Hash the cluster file to prove parity. |
| AnySat | filename_match_unverified | g-astruc/AnySat@63f252119bab | Replace the floating main URL and hash the cluster cache. |
| Prithvi | filename_match_unverified | ibm-nasa-geospatial/Prithvi-100M@f3a9ea7a1723 | Hash the cluster file to prove parity. |
| SatlasNet | filename_match_unverified | allenai/satlas-pretrain@b1f8de04dacf | Hash the cluster file to prove parity. |
| TerraFM | filename_match_unverified | MBZUAI/TerraFM@3631173fec2c | Hash the cluster file to prove parity. |
| DINOv3 | unresolved | facebook/dinov3-vitb16-pretrain-lvd1689m@5931719e67bb | Classification omitted revision; dense tasks used an unverified local converted checkpoint. Hash both and verify conversion parity. |
| Panopticon | unresolved | lewaldm/panopticon@c8c2bb955581 | No Panopticon loader/config exists on public main. Recover the executed integration and hash its cached weight. |
| TerraMind | filename_match_unverified | ibm-esa-geospatial/TerraMind-1.0-base@fb96c70d0a5f | Recover exact package versions and hash the cluster cache. Ten live spreadsheet cells do not match the historical export. |

