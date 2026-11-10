#include "three_center.h"
#include <iostream>

int main() {
  try {
    using synkit::stereo::isomorphism;
    for (bool aldehyde : {false, true}) {
      const auto result = isomorphism(three_center::molecule("LRL", aldehyde),
                                      three_center::molecule("RLR", aldehyde));
      if (!result.complete()) throw std::runtime_error("Example search stopped");
      std::cout << (aldehyde ? "CHO" : "CH2OH") << "-(CHOH)3-CH2OH: LRL vs RLR = "
                << (result.isomorphic() ? "same" : "different") << '\n';
      if (result.isomorphic())
        std::cout << "  site mapping: A1 -> B" << result.mapping[1]
                  << ", A2 -> B" << result.mapping[2] << ", A3 -> B" << result.mapping[3] << '\n';
    }
    auto middle = isomorphism(three_center::molecule("LRL", true),
                              three_center::molecule("LLL", true));
    std::cout << "CHO-(CHOH)3-CH2OH: LRL vs LLL (middle change) = "
              << (middle.isomorphic() ? "same" : "different") << '\n';
    std::cout << "8 x 8 exact comparison matrices (1 = same):\n";
    for (bool aldehyde : {false, true}) {
      std::cout << (aldehyde ? "aldehyde" : "symmetric") << '\n';
      for (const auto &a : three_center::words) {
        std::cout << a << ' ';
        for (const auto &b : three_center::words)
          std::cout << isomorphism(three_center::molecule(a, aldehyde),
                                   three_center::molecule(b, aldehyde)).isomorphic();
        std::cout << '\n';
      }
    }
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
