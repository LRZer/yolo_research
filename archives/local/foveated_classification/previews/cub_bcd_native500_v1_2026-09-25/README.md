# CUB B/C/D 实际图片预览（仅官方训练图片）

本目录仅用于用户检查图像处理效果；没有训练分类模型、没有改动原始图片或标注，也没有使用官方测试图片。所有 A/B/C/D 单张 PNG 都是真实 500×500 模型候选输入，联系表只是并排展示。

训练集有效像素 RGB 均值（0–255）：123.978, 127.621, 110.508。画布填充使用四舍五入的该颜色。

B：整幅有效图 σ=8×(原图长边/500) 模糊；C：真值框外扩10%内清晰，框外按70×(长边/500)过渡至均值色；D：同一保护区外按0/2/4/8×(长边/500)多尺度高斯连续混合。红框=原始框裁边，青框=保护框；框线只在 *_boxes.png，绝不叠在模型候选输入图上。

| 用途 | 类别/ID | 尺寸 | 框面积 | 对比图 | 标注示意 |
|---|---|---:|---:|---|---|
| wide_small_bird | 173.Orange_crowned_Warbler / 10140 | 500×333 | 14.0% | [A/B/C/D](10140_wide_small_bird/A_B_C_D_contact_sheet.png) | [框](10140_wide_small_bird/A_original_boxes.png) |
| wide_medium_bird | 045.Northern_Fulmar / 2568 | 500×369 | 32.0% | [A/B/C/D](02568_wide_medium_bird/A_B_C_D_contact_sheet.png) | [框](02568_wide_medium_bird/A_original_boxes.png) |
| wide_large_bird | 107.Common_Raven / 6223 | 500×333 | 58.0% | [A/B/C/D](06223_wide_large_bird/A_B_C_D_contact_sheet.png) | [框](06223_wide_large_bird/A_original_boxes.png) |
| portrait | 082.Ringed_Kingfisher / 4786 | 317×500 | 32.0% | [A/B/C/D](04786_portrait/A_B_C_D_contact_sheet.png) | [框](04786_portrait/A_original_boxes.png) |
| square | 158.Bay_breasted_Warbler / 9234 | 500×500 | 31.8% | [A/B/C/D](09234_square/A_B_C_D_contact_sheet.png) | [框](09234_square/A_original_boxes.png) |
| small_native_image | 032.Mangrove_Cuckoo / 1767 | 300×288 | 31.7% | [A/B/C/D](01767_small_native_image/A_B_C_D_contact_sheet.png) | [框](01767_small_native_image/A_original_boxes.png) |
| out_of_bounds_box | 029.American_Crow / 1592 | 357×500 | 30.9% | [A/B/C/D](01592_out_of_bounds_box/A_B_C_D_contact_sheet.png) | [框](01592_out_of_bounds_box/A_original_boxes.png) |
| grayscale_jpg | 063.Ivory_Gull / 3617 | 500×326 | 32.7% | [A/B/C/D](03617_grayscale_jpg/A_B_C_D_contact_sheet.png) | [框](03617_grayscale_jpg/A_original_boxes.png) |
