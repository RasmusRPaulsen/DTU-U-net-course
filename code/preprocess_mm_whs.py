import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def read_file_and_print_info():
    full_name = 'C:/data/MM-WHS/ct_train_test/ct_train/ct_train_1001_image.nii.gz'
    image = sitk.ReadImage(full_name)
    print(f"Origin: {image.GetOrigin()}")
    print(f"Size: {image.GetSize()}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Width: {image.GetWidth()}")
    print(f"Height: {image.GetHeight()}")
    print(f"Depth: {image.GetDepth()}")
    print(f"PixelID: {image.GetPixelID()}")
    print(f"Number of components per pixel: {image.GetNumberOfComponentsPerPixel()}")
    print(f"Image Direction: {image.GetDirection()}")


def read_file_and_show_a_slice():
    full_name = 'C:/data/MM-WHS/ct_train_test/ct_train/ct_train_1001_image.nii.gz'
    image = sitk.ReadImage(full_name)

    # Convert to numpy array
    img_np = sitk.GetArrayFromImage(image)
    print(f"SITK image size: {image.GetSize()}")
    print(f"NP image shape: {img_np.shape}")

    # Convert numpy array from (z, y, x) to (x, y, z)
    img_np = img_np.transpose(2, 1, 0)
    print(f"NP image shape after transposing: {img_np.shape}")

    slice_number = 100
    one_slice = img_np[:, :, slice_number]
    print(f"NP slice shape: {one_slice.shape}")
    plt.imshow(one_slice, vmin=-100, vmax=500, cmap="gray")
    plt.show()


def read_label_and_show_a_slice():
    full_name = 'C:/data/MM-WHS/ct_train_test/ct_train/ct_train_1001_label.nii.gz'
    image = sitk.ReadImage(full_name)

    print(f"PixelID: {image.GetPixelID()}")
    print(f"Number of components per pixel: {image.GetNumberOfComponentsPerPixel()}")

    # Convert to numpy array
    img_np = sitk.GetArrayFromImage(image)
    print(f"SITK image size: {image.GetSize()}")
    print(f"NP image shape: {img_np.shape}")

    # Convert numpy array from (z, y, x) to (x, y, z)
    img_np = img_np.transpose(2, 1, 0)
    print(f"NP image shape after transposing: {img_np.shape}")

    slice_number = 100
    one_slice = img_np[:, :, slice_number]
    print(f"NP slice shape: {one_slice.shape}")
    # plt.imshow(one_slice, cmap="gray")
    plt.imshow(one_slice)
    plt.show()

    binary_slice = one_slice > 0
    plt.imshow(binary_slice)
    plt.show()

    heart_sum = np.sum(binary_slice)
    is_there_a_heart = False
    if heart_sum > 0:
        is_there_a_heart = True

    print(f"Heart sum {heart_sum} so is there a heart {is_there_a_heart}")


def resample_image():
    full_name = 'C:/data/MM-WHS/ct_train_test/ct_train/ct_train_1001_image.nii.gz'
    out_name =  'C:/data/test/MM-WHS-1001-resampled.nii.gz'
    image = sitk.ReadImage(full_name)

    # how many voxel per slice side
    desired_num_voxels = 64

    current_n_vox = image.GetWidth()
    # With what factor are we scaling?
    sample_factor = float(current_n_vox / desired_num_voxels)

    current_spacing = image.GetSpacing()
    # What will be the new in-slice spacing
    new_spacing = current_spacing[0] * sample_factor

    print(f"Old spacing: {current_spacing } new spacing: {new_spacing}")

    # voxel size in the direction of the patient
    depth_spacing = current_spacing[2]
    n_vox_depth = image.GetDepth()

    new_n_vox_depth = int(n_vox_depth * depth_spacing / new_spacing)

    new_volume_size = [desired_num_voxels, desired_num_voxels, new_n_vox_depth]

    # Create new image with desired properties
    new_image = sitk.Image(new_volume_size, image.GetPixelIDValue())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing([new_spacing, new_spacing, new_spacing])
    new_image.SetDirection(image.GetDirection())

    # Make translation with no offset, since sitk.Resample needs this arg.
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((0, 0, 0))

    interpolator = sitk.sitkLinear
    # Create final reaampled image
    resampled_image = sitk.Resample(image, new_image, translation, interpolator)

    sitk.WriteImage(resampled_image, out_name)


def resample_image_and_label_image():
    full_name = 'C:/data/MM-WHS/ct_train_test/ct_train/ct_train_1001_image.nii.gz'
    out_name = 'C:/data/test/MM-WHS/MM-WHS-1001-resampled_intensity.nii.gz'
    label_name = 'C:/data/MM-WHS/ct_train_test/ct_train/ct_train_1001_label.nii.gz'
    label_out_name = 'C:/data/test/MM-WHS/MM-WHS-1001-label-resampled.nii.gz'

    image = sitk.ReadImage(full_name)
    label_image = sitk.ReadImage(label_name, sitk.sitkUInt16)
    # print(f"PixelID: {label_image.GetPixelID()}")
    # print(f"PixelID: {label_image.GetPixelIDTypeAsString ()}")

    # how many voxel per slice side
    desired_num_voxels = 64

    current_n_vox = image.GetWidth()
    # With what factor are we scaling?
    sample_factor = float(current_n_vox / desired_num_voxels)

    current_spacing = image.GetSpacing()
    # What will be the new in-slice spacing
    new_spacing = current_spacing[0] * sample_factor

    print(f"Old spacing: {current_spacing } new spacing: {new_spacing}")

    # voxel size in the direction of the patient
    depth_spacing = current_spacing[2]
    n_vox_depth = image.GetDepth()

    new_n_vox_depth = int(n_vox_depth * depth_spacing / new_spacing)

    new_volume_size = [desired_num_voxels, desired_num_voxels, new_n_vox_depth]

    # Create new image with desired properties
    new_image = sitk.Image(new_volume_size, image.GetPixelIDValue())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing([new_spacing, new_spacing, new_spacing])
    new_image.SetDirection(image.GetDirection())

    # Make translation with no offset, since sitk.Resample needs this arg.
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((0, 0, 0))

    # https://simpleitk.readthedocs.io/en/v1.2.4/Documentation/docs/source/fundamentalConcepts.html#resampling
    # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/21_Transforms_and_Resampling.html
    interpolator = sitk.sitkLinear
    # Create final resampled image
    resampled_image = sitk.Resample(image, new_image, translation, interpolator)

    # Resample intensities so HU [-100,500] -> [0, 1]
    intensity_resampled_image = sitk.Threshold(resampled_image,
                                lower=-2000,
                                upper=500,
                                outsideValue=500)
    intensity_resampled_image = sitk.Threshold(intensity_resampled_image,
                                lower=-100,
                                upper=2000,
                                outsideValue=-100)
    intensity_resampled_image = sitk.RescaleIntensity(intensity_resampled_image,
                                       outputMinimum=0,
                                       outputMaximum=1)

    sitk.WriteImage(intensity_resampled_image, out_name)

    # https://examples.itk.org/src/core/imagefunction/resamplesegmentedimage/documentation
    # https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html
    # Resample label image
    new_label_image = sitk.Image(new_volume_size, label_image.GetPixelIDValue())
    new_label_image.SetOrigin(image.GetOrigin())
    new_label_image.SetSpacing([new_spacing, new_spacing, new_spacing])
    new_label_image.SetDirection(image.GetDirection())

    # interpolator = sitk.sitkLabelGaussian
    interpolator = sitk.sitkNearestNeighbor
    resampled_image = sitk.Resample(label_image, new_label_image, translation, interpolator)

    sitk.WriteImage(resampled_image, label_out_name)


def sample_slices_from_resampled_volume(mm_whs_id):
    resample_name = f'C:/data/test/MM-WHS/MM-WHS-{mm_whs_id}-resampled_intensity.nii.gz'
    label_name = f'C:/data/test/MM-WHS/MM-WHS-{mm_whs_id}-label-resampled.nii.gz'
    out_name =  'C:/data/test/MM-WHS/'

    image = sitk.ReadImage(resample_name)
    labels = sitk.ReadImage(label_name)

    # Convert to numpy array
    img_np = sitk.GetArrayFromImage(image)
    print(f"SITK image size: {image.GetSize()}")

    # Convert numpy array from (z, y, x) to (x, y, z)
    img_np = img_np.transpose(2, 1, 0)

    labels_np = sitk.GetArrayFromImage(labels)
    labels_np = labels_np.transpose(2, 1, 0)

    n_slices = labels_np.shape[2]
    print(f"n slices {n_slices}")

    for i in range(n_slices):
        slice_number = i
        img_slice = img_np[:, :, slice_number]
        lab_slice = labels_np[:, :, slice_number]

        bin_slice = lab_slice > 0
        has_heart = "no_heart"
        if np.sum(bin_slice) > 0:
            has_heart = "has_heart"

        out_name_img_slice = f"{out_name}/MM-WHS-{mm_whs_id}-slice-{slice_number}-image-{has_heart}.npy"
        out_name_img_label = f"{out_name}/MM-WHS-{mm_whs_id}-slice-{slice_number}-label-{has_heart}.npy"

        np.save(out_name_img_slice, img_slice)
        np.save(out_name_img_label, lab_slice)


def write_all_slices():
    mm_whs_id = "1001"
    sample_slices_from_resampled_volume(mm_whs_id)


# Check if we have saved slices correctly
def debug_check_single_slice():
    out_name =  'C:/data/test/MM-WHS/'
    mm_whs_id = "1001"
    has_heart = "has_heart"
    i = 30

    name_img_slice = f"{out_name}/MM-WHS-{mm_whs_id}-slice-{i}-image-{has_heart}.npy"
    name_img_label = f"{out_name}/MM-WHS-{mm_whs_id}-slice-{i}-label-{has_heart}.npy"
    one_slice = np.load(name_img_slice)
    labels = np.load(name_img_label)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(one_slice, cmap="gray")
    ax1.set_title('Rescaled image')
    ax1.axis('off')
    ax2.imshow(labels)
    ax2.set_title('Corresponding labels')
    ax2.axis('off')
    plt.show()


if __name__ == '__main__':
    # read_label_and_show_a_slice()
    # read_file_and_print_info()
    # read_file_and_show_a_slice()
    # resample_image()
    # resample_image_and_label_image()
    write_all_slices()
    debug_check_single_slice()
