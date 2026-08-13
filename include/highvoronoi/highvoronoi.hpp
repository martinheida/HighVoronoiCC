
#pragma once

#include <cstddef>
#include <string_view>

// ============================================================================
// Export-Makro
// ============================================================================
//
// Unter Linux ist für diesen einfachen Startfall kein spezielles Attribut
// nötig. Unter Windows sorgt das Makro dafür, dass kompilierte Funktionen bei
// einer DLL korrekt exportiert/importiert werden.
//
#if defined(_WIN32) && defined(HIGHVORONOI_SHARED_LIBRARY)
    #if defined(HIGHVORONOI_BUILDING_LIBRARY)
        #define HIGHVORONOI_API __declspec(dllexport)
    #else
        #define HIGHVORONOI_API __declspec(dllimport)
    #endif
#else
    #define HIGHVORONOI_API
#endif


// ============================================================================
// Öffentliche Template-/Modul-Header
// ============================================================================
//
// Der Benutzer soll nur
//
//     #include <highvoronoi/highvoronoi.hpp>
//
// schreiben müssen.
//
// Neue Template-Komponenten können in eigenen Headern organisiert werden und
// werden hier eingebunden. Die Definition eines Templates muss normalerweise
// für den Compiler sichtbar sein, deshalb gehört Template-Code in Header.
//
// Beispiel:
#include <highvoronoi/geometry/point.hpp>
#include <highvoronoi/geometry/voronoi_nodes.hpp>
#include <highvoronoi/geometry/compute_voronoi.hpp>
#include <highvoronoi/geometry/mesh_validation.hpp>
#include <highvoronoi/parameters.hpp>


#include <highvoronoi/templates/example.hpp>
// Should be private:
//#include <src/detail/locks.hpp>

namespace highvoronoi {

// ============================================================================
// Nicht-template API
// ============================================================================
//
// Solche Funktionen werden in src/highvoronoi.cpp implementiert und beim
// Bau der Bibliothek tatsächlich kompiliert.
//
[[nodiscard]]
HIGHVORONOI_API std::string_view version() noexcept;


// Ein zweites minimales Beispiel für kompilierten Code.
[[nodiscard]]
HIGHVORONOI_API std::size_t compiled_example(std::size_t value) noexcept;

} // namespace highvoronoi
