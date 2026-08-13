
#pragma once

namespace highvoronoi {

// ============================================================================
// Template-Beispiel
// ============================================================================
//
// Template-Funktionen werden im Header vollständig definiert.
// Es gibt dafür normalerweise KEINE separate .cpp-Datei.
//
template <class T>
[[nodiscard]]
constexpr T square(const T& value)
{
    return value * value;
}

} // namespace highvoronoi
