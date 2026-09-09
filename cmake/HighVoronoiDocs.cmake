# HighVoronoi documentation target.
# Included only when HIGHVORONOI_BUILD_DOCS=ON.
#
# Hard rule: documentation inputs are explicitly controlled. In addition to
# the manual/example Markdown whitelist, only *.hpp files below
# include/highvoronoi are admitted as API/source documentation.

find_package(Doxygen REQUIRED)

set(
    HIGHVORONOI_DOCS_OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/manual"
    CACHE PATH
    "Output directory for generated HighVoronoi documentation"
)

set(HIGHVORONOI_DOCS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs")
set(
    HIGHVORONOI_ARCHITECTURE_DIR
    "${HIGHVORONOI_DOCS_DIR}/architecture and scientific foundation"
)
set(
    HIGHVORONOI_CONCEPT_PAPER
    "${HIGHVORONOI_ARCHITECTURE_DIR}/HVConceptPaper.pdf"
)
set(HIGHVORONOI_LICENSE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

# Explicit user-manual whitelist. docs/for AI agents is intentionally absent.
set(HIGHVORONOI_MANUAL_PAGES
    "${HIGHVORONOI_DOCS_DIR}/index.md"
    "${HIGHVORONOI_DOCS_DIR}/getting_started.md"
    "${HIGHVORONOI_DOCS_DIR}/level_1_3_api_comparison.md"
    "${HIGHVORONOI_DOCS_DIR}/ordinary_voronoi.md"
    "${HIGHVORONOI_DOCS_DIR}/accessing_results.md"
    "${HIGHVORONOI_DOCS_DIR}/incremental.md"
    "${HIGHVORONOI_DOCS_DIR}/high_voronoi.md"
    "${HIGHVORONOI_DOCS_DIR}/spherical.md"
    "${HIGHVORONOI_DOCS_DIR}/integration.md"
    "${HIGHVORONOI_DOCS_DIR}/integration_level_1_3_api_comparison.md"
    "${HIGHVORONOI_DOCS_DIR}/parallelism.md"
    "${HIGHVORONOI_DOCS_DIR}/configuration.md"
    "${HIGHVORONOI_DOCS_DIR}/validation.md"
    "${HIGHVORONOI_DOCS_DIR}/examples_and_tests.md"
    "${HIGHVORONOI_DOCS_DIR}/test_overview_developers.md"
    "${HIGHVORONOI_DOCS_DIR}/api_reference.md"
    "${HIGHVORONOI_ARCHITECTURE_DIR}/ARCHITECTURE.md"
)


# Enumerate HighVoronoi headers explicitly. Doxygen never receives the include
# directory itself, so Markdown or any other file type below include/ cannot be
# discovered accidentally. The bundled nanoflann implementation is third-party
# code and is deliberately excluded from the HighVoronoi API indexes.
file(GLOB_RECURSE HIGHVORONOI_API_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/highvoronoi/*.hpp"
)
list(FILTER HIGHVORONOI_API_HEADERS EXCLUDE REGEX
    "/include/highvoronoi/search/detail/nanoflann\\.hpp$"
)
list(SORT HIGHVORONOI_API_HEADERS)

# Explicit Markdown files from examples/. No .cpp or other example files are
# documentation inputs.
set(HIGHVORONOI_EXAMPLE_PAGES
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/highvoronoi_examples_README.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/incremental/README.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/parallel/README.md"
)

set(HIGHVORONOI_REQUIRED_DOCS
    ${HIGHVORONOI_MANUAL_PAGES}
    ${HIGHVORONOI_EXAMPLE_PAGES}
    ${HIGHVORONOI_API_HEADERS}
    "${HIGHVORONOI_CONCEPT_PAPER}"
    "${HIGHVORONOI_LICENSE_SOURCE}"
)

foreach(required_doc IN LISTS HIGHVORONOI_REQUIRED_DOCS)
    if(NOT EXISTS "${required_doc}")
        message(FATAL_ERROR
            "Required HighVoronoi documentation input not found: ${required_doc}")
    endif()
endforeach()

# Render the repository's single authoritative LICENSE file as a Doxygen page
# without maintaining a second copy of the license text in docs/.
set(HIGHVORONOI_GENERATED_DOCS_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs_generated")
set(HIGHVORONOI_LICENSE_PAGE "${HIGHVORONOI_GENERATED_DOCS_DIR}/LICENSE.md")
file(MAKE_DIRECTORY "${HIGHVORONOI_GENERATED_DOCS_DIR}")
file(READ "${HIGHVORONOI_LICENSE_SOURCE}" HIGHVORONOI_LICENSE_TEXT)
file(WRITE "${HIGHVORONOI_LICENSE_PAGE}"
"@page highvoronoi_license License\n\n"
"@verbatim\n"
"${HIGHVORONOI_LICENSE_TEXT}\n"
"@endverbatim\n")

# Build one quoted Doxygen INPUT line from controlled explicit files only.
set(HIGHVORONOI_DOXYGEN_INPUT "")
foreach(doc IN LISTS
        HIGHVORONOI_MANUAL_PAGES
        HIGHVORONOI_EXAMPLE_PAGES
        HIGHVORONOI_API_HEADERS)
    string(APPEND HIGHVORONOI_DOXYGEN_INPUT " \"${doc}\"")
endforeach()
string(APPEND HIGHVORONOI_DOXYGEN_INPUT " \"${HIGHVORONOI_LICENSE_PAGE}\"")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    @ONLY
)

# Every invocation starts from an empty HTML directory, so removed pages cannot
# survive as stale output.
add_custom_target(docs
    COMMAND
        "${CMAKE_COMMAND}" -E rm -rf
        "${HIGHVORONOI_DOCS_OUTPUT_DIR}/html"
    COMMAND
        "${DOXYGEN_EXECUTABLE}"
        "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    COMMAND
        "${CMAKE_COMMAND}" -E make_directory
        "${HIGHVORONOI_DOCS_OUTPUT_DIR}/html/architecture and scientific foundation"
    COMMAND
        "${CMAKE_COMMAND}" -E copy_if_different
        "${HIGHVORONOI_CONCEPT_PAPER}"
        "${HIGHVORONOI_DOCS_OUTPUT_DIR}/html/architecture and scientific foundation/HVConceptPaper.pdf"
    WORKING_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT
        "Generating HighVoronoi manual and header API indexes from controlled inputs"
    VERBATIM
)

message(STATUS
    "HighVoronoi documentation enabled: ${HIGHVORONOI_DOCS_OUTPUT_DIR}/html/index.html")
