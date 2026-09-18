#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <iostream>

using ninfer::targets::qwen3_6_27b::Package;
using namespace ninfer::targets::qwen3_6_27b::detail;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: load_verify <model.ninfer>\n";
        return 1;
    }
    try {
        std::cout << "Opening " << argv[1] << "...\n";
        ninfer::artifact::Reader reader(argv[1]);
        std::cout << "Reader parsed OK! Identity: " << reader.identity().model_id 
                  << " / " << reader.identity().weights_id << "\n";
        std::cout << "Total objects: " << reader.objects().size() << "\n";
        
        const auto profile = Package::resolve_weights(reader.identity());
        ninfer::artifact::Binder binder(reader);
        ninfer::targets::qwen3_6::StartupFeatures features{
            .vision = true,
            .speculative = ninfer::SpeculativeBackend::Mtp,
            .proposal_head = ninfer::ProposalHead::Optimized,
        };
        const auto plan = bind_artifact(binder, profile, features);
        std::cout << "Bind plan completed successfully with VISION + MTP enabled!\n";
        std::cout << "  device_objects: " << plan.materialization.device_objects.size() << "\n";
        std::cout << "  host_objects: " << plan.materialization.host_objects.size() << "\n";
        std::cout << "  device_capacity_bytes: " << plan.materialization.device_capacity_bytes << "\n";
        std::cout << "ALL TENSOR ALIGNMENTS AND RUNTIME BINDINGS (VISION+MTP) VERIFIED SUCCESSFULLY!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "LOAD ERROR: " << e.what() << "\n";
        return 2;
    }
}
