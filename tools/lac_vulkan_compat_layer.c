/*
 * lac_compat_layer.c  —  v2 (simplified)
 *
 * Minimal Vulkan intercept layer for LAC simulator on WSL2.
 *
 * Problem: Mesa dzn (D3D12 translation layer) reports conformanceVersion=0.0.0.0.
 * UE4 4.26 rejects ALL devices with zero conformance version, producing the
 * "Incompatible Vulkan driver found!" dialog. VK_EXT_robustness2 is optional in
 * UE4 4.26 (enables nullDescriptor) — it is NOT the rejection reason.
 *
 * Fix: intercept vkGetPhysicalDeviceProperties2 and patch conformanceVersion
 * from 0.0.0.0 → 1.3.0.0 for any device that reports all-zero conformance.
 * Nothing else is intercepted — no vkCreateDevice, no extension list surgery.
 *
 * Compile:
 *   gcc -shared -fPIC -fvisibility=hidden -Wall \
 *       -o /tmp/liblac_compat.so /tmp/lac_compat_layer.c -I/usr/include -ldl
 *
 * Enable:
 *   export VK_LAYER_PATH=/tmp
 *   export VK_INSTANCE_LAYERS=VK_LAYER_lac_compat
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <string.h>
#include <stdio.h>

/* ── Saved next-layer pointers ──────────────────────────────────────────── */
static PFN_vkGetInstanceProcAddr            next_GetInstanceProcAddr  = NULL;
static PFN_vkGetPhysicalDeviceProperties2   next_GetPhysDevProps2     = NULL;

/* ── Patch conformanceVersion in pNext chain ────────────────────────────── */
static void patch_conformance(void *pNext_head)
{
    VkBaseOutStructure *s = (VkBaseOutStructure *)pNext_head;
    while (s) {
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES) {
            VkPhysicalDeviceVulkan12Properties *p =
                (VkPhysicalDeviceVulkan12Properties *)s;
            if (p->conformanceVersion.major == 0 &&
                p->conformanceVersion.minor == 0 &&
                p->conformanceVersion.subminor == 0 &&
                p->conformanceVersion.patch == 0)
            {
                fprintf(stderr,
                    "[lac_compat] patching Vulkan12Properties conformanceVersion "
                    "0.0.0.0 -> 1.3.0.0\n");
                p->conformanceVersion.major    = 1;
                p->conformanceVersion.minor    = 3;
                p->conformanceVersion.subminor = 0;
                p->conformanceVersion.patch    = 0;
            }
        }
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES) {
            VkPhysicalDeviceDriverProperties *p =
                (VkPhysicalDeviceDriverProperties *)s;
            if (p->conformanceVersion.major == 0 &&
                p->conformanceVersion.minor == 0 &&
                p->conformanceVersion.subminor == 0 &&
                p->conformanceVersion.patch == 0)
            {
                fprintf(stderr,
                    "[lac_compat] patching DriverProperties conformanceVersion "
                    "0.0.0.0 -> 1.3.0.0\n");
                p->conformanceVersion.major    = 1;
                p->conformanceVersion.minor    = 3;
                p->conformanceVersion.subminor = 0;
                p->conformanceVersion.patch    = 0;
            }
        }
        s = s->pNext;
    }
}

static VKAPI_ATTR void VKAPI_CALL
lac_vkGetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice,
                                   VkPhysicalDeviceProperties2 *pProperties)
{
    next_GetPhysDevProps2(physicalDevice, pProperties);
    patch_conformance(pProperties->pNext);
}

/* ── Instance creation — grab the next-layer proc addr ─────────────────── */
static VKAPI_ATTR VkResult VKAPI_CALL
lac_vkCreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkInstance *pInstance)
{
    /* Walk pNext to find loader's chain info and extract next layer's gipa */
    VkLayerInstanceCreateInfo *chain =
        (VkLayerInstanceCreateInfo *)pCreateInfo->pNext;
    while (chain &&
           !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
             chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerInstanceCreateInfo *)chain->pNext;

    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;

    next_GetInstanceProcAddr =
        chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateInstance create_fn =
        (PFN_vkCreateInstance)next_GetInstanceProcAddr(
            VK_NULL_HANDLE, "vkCreateInstance");

    /* Advance the chain BEFORE calling down */
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    VkResult res = create_fn(pCreateInfo, pAllocator, pInstance);
    if (res != VK_SUCCESS) return res;

    /* Cache the physical-device-properties function from the next layer */
    next_GetPhysDevProps2 =
        (PFN_vkGetPhysicalDeviceProperties2)
        next_GetInstanceProcAddr(*pInstance, "vkGetPhysicalDeviceProperties2");

    return VK_SUCCESS;
}

/* ── Layer dispatch — only override what we need ────────────────────────── */
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
lac_vkGetInstanceProcAddr(VkInstance instance, const char *pName)
{
    if (strcmp(pName, "vkCreateInstance") == 0)
        return (PFN_vkVoidFunction)lac_vkCreateInstance;
    if (strcmp(pName, "vkGetInstanceProcAddr") == 0)
        return (PFN_vkVoidFunction)lac_vkGetInstanceProcAddr;
    if (strcmp(pName, "vkGetPhysicalDeviceProperties2") == 0)
        return (PFN_vkVoidFunction)lac_vkGetPhysicalDeviceProperties2;

    if (!next_GetInstanceProcAddr) return NULL;
    return next_GetInstanceProcAddr(instance, pName);
}

/* ── Loader negotiation ──────────────────────────────────────────────────── */
__attribute__((visibility("default")))
VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct)
{
    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->loaderLayerInterfaceVersion      = 2;
        pVersionStruct->pfnGetInstanceProcAddr           = lac_vkGetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr             = NULL;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr     = NULL;
    }
    return VK_SUCCESS;
}
