
#----------- path -----------#
sys.path.append(str(Path.cwd().resolve().parent.parent))
path_assets = path.get_assets_path() # /home/nii/Documents/SGU_2025_ML-Fundamentals/assets
path_assets_images_mushroom = path_assets / "images" / "mushroom"
print(f"path_assets_images_mushroom: {path_assets_images_mushroom}")
#--- image:
img_1_1_mushroom_structure = path_assets_images_mushroom / "mushroom-structure.jpg"
img_1_2_mushroom_cap_shape = path_assets_images_mushroom / "mushroom-cap-shape.jpg"
img_1_3_mushroom_cap_surface = path_assets_images_mushroom / "mushroom-cap-surface-768x138.jpg"
img_1_4_mushroom_gill_attachment = path_assets_images_mushroom / "mushroom-gill-attachment-768x146.jpg"
img_1_5_mushroom_gill_spacing = path_assets_images_mushroom / "mushroom-gill-spacing.jpg"
img_1_6_mushroom_gill_tissue_arrangement = path_assets_images_mushroom / "mushroom-gill-tissue-arrangement.jpg"
img_1_7_mushroom_stalk = path_assets_images_mushroom / "mushroom-stalk-768x259.jpg" 
img_1_8_mushroom_ring_type = path_assets_images_mushroom / "mushroom-ring-type-768x207.jpg"
# mushroom-cap-shape.jpg                mushroom-gill-spacing.jpg             mushroom-stalk-768x259.jpg
# mushroom-cap-surface-768x138.jpg      mushroom-gill-tissue-arrangement.jpg  mushroom-structure.jpg
# mushroom-gill-attachment-768x146.jpg  mushroom-ring-type-768x207.jpg
