# Params-Handbuch

## 1. `DataBaseParams`

```cpp
DataBaseParams<
    Scalar,
    Index,
    HashGenerator,
    ContainerMode,
    QueueTable
>
```

### `Scalar`

Bestimmt den Fließkommatyp in der Datenbank wie `float` oder `double` der eigentlichen geometrischen Daten.

### `Index`

Bestimmt den Integer-Typ für Indizes in der Datenbank, Zellnummern und ähnliche Referenzen. `std::uint32_t`, `td::uint64_t`, `std::int64_t`

### `HashGenerator`

Bestimmt:

1. wie aus einem Key der gespeicherte Hash-Fingerprint erzeugt wird,
2. wie der erste Tabellenindex berechnet wird,
3. wie die weiteren Probe-Indizes berechnet werden.

Beispielsweise kann ein Generator zwei Murmur-Werte für das Double Hashing verwenden und zusätzlich einen XXHash-Wert zur Kollisionskontrolle speichern. 

### `QueueTable`

Bestimmt den Algorithmus **einer einzelnen Queue-Hash-Tabelle**.

Es gibt zwei Varianten:

#### Alte Tabelle
--- hier konkreter Typ
* ein globaler Lock pro Operation,
* einfache Implementierung,
* Lesen und Schreiben werden stärker serialisiert,
* vor allem als Referenz oder einfache Single-Thread-Variante sinnvoll.

#### Tabelle 2
--- hier konkreter Typ
* zählt die tatsächlich enthaltenen Einträge,
* besitzt einen getrennten Writer-Lock,
* Leser dürfen während des Aufbaus einer neuen Tabelle weiter auf der alten Tabelle lesen,
* nur der abschließende Austausch der Tabellen blockiert kurz alle Zugriffe,
* erweitert typischerweise, sobald mehr als die Hälfte der Plätze real belegt ist.

**Standardmäßig sollte Tabelle 2 verwendet werden.**

---

## 2. `ContainerMode`

Der Container-Modus entscheidet, ob eine oder mehrere Hash-Tabellen erzeugt werden.

### `DirectHash`

```cpp
DirectHash{hash_capacity}
```

Erzeugt:

```text
eine einzelne Hash-Tabelle
```

Alle Keys landen in derselben Tabelle.

`hash_capacity` ist die anfängliche Zahl der Plätze.

Geeignet für:

* kleine Datenmengen,
* Single Thread,
* Fälle ohne starke Lock-Konkurrenz,
* Situationen, in denen keine Aufteilung nach `key[0]` benötigt wird.

---

### `StaticHash<N>`

```cpp
StaticHash<N>{hash_capacity}
```

Erzeugt:

```text
ein festes Array aus N unabhängigen Hash-Tabellen
```

Die Tabelle wird ausgewählt durch:

```cpp
table = key[0] % N;
```

Jede der `N` Tabellen startet mit `hash_capacity` Plätzen.

Eigenschaften:

* `N` steht zur Compilezeit fest,
* der Container wird niemals verlängert,
* kein zusätzlicher Container-Lock ist erforderlich,
* Zugriffe auf verschiedene Teil-Tabellen können unabhängig voneinander laufen.

Geeignet für:

* bekannte, feste Zahl von Partitionen,
* gleichmäßig verteilte erste Key-Werte,
* parallelen Code mit möglichst wenig Lock-Konkurrenz.

Heuristisch:

```text
größeres N
    → weniger Konkurrenz pro Tabelle
    → mehr einzelne Tabellen und mehr Grundspeicher
```

---

### `DynamicHash`

```cpp
DynamicHash<>{
    initial_table_count,
    block_size,
    hash_capacity
}
```

Erzeugt:

```text
einen dynamisch wachsenden Vector unabhängiger Hash-Tabellen
```

Die Tabelle wird ausgewählt durch:

```cpp
table = key[0] / block_size;
```

Beispiele bei `block_size = 1000`:

```text
key[0] =    0 ...  999  → Tabelle 0
key[0] = 1000 ... 1999  → Tabelle 1
key[0] = 2000 ... 2999  → Tabelle 2
```

Falls die benötigte Tabelle noch nicht existiert, wird der Vector verlängert.

Parameter:

```text
initial_table_count
    anfängliche Zahl bereits erzeugter Tabellen

block_size
    Zahl aufeinanderfolgender key[0]-Werte pro Tabelle

hash_capacity
    anfängliche Kapazität jeder einzelnen Tabelle
```

Der Vector besitzt einen eigenen Read/Write-Lock:

* normale Zugriffe nehmen nur einen Read-Lock auf die Vector-Struktur,
* nur eine Verlängerung des Vectors benötigt den Write-Lock,
* die einzelnen Tabellen regeln ihre eigene Synchronisation selbst.

Geeignet für:

* unbekannten oder wachsenden Wertebereich von `key[0]`,
* natürliche Aufteilung in Werteblöcke,
* Vermeidung eines sehr großen statischen Arrays.

Heuristisch:

```text
kleiner block_size
    → mehr Teil-Tabellen
    → weniger Konkurrenz pro Tabelle
    → mehr Verwaltungs- und Grundspeicher

großer block_size
    → weniger Teil-Tabellen
    → mehr Einträge und Konkurrenz pro Tabelle
```

Eine vernünftige Anfangsschätzung ist:

```cpp
initial_table_count
    ≈ erwarteter_maximaler_key0 / block_size + 1;
```

Der Vector kann jedoch später weiter wachsen.

---

## 3. Effekt von `DataBaseParams`

Die Kombination

```cpp
DataBaseParams<
    Scalar,
    Index,
    HashGenerator,
    ContainerMode,
    QueueTable
>
```

erzeugt intern effektiv:

```text
ContainerMode
    └── enthält eine oder mehrere QueueTable-Tabellen
            └── jede Tabelle verwendet HashGenerator
                    └── jeder Zugriff verwendet den später gewählten Lock
```

Der Lock steht nicht in `DataBaseParams`. Er wird später durch

```cpp
SingleThread
```

oder

```cpp
MultiThread
```

festgelegt.

```text
SingleThread
    → EmptyLock
    → keine reale Synchronisation

MultiThread
    → ReadWriteLock
    → synchronisierte Zugriffe
```

---

## 4. `EdgeBufferParams`

```cpp
EdgeBufferParams<
    HashGenerator,
    ContainerMode,
    EdgeTable
>
```

Die Bedeutung ist dieselbe wie bei der Queue-Konfiguration, nur ohne `Scalar` und `Index`.

Es wird effektiv erzeugt:

```text
ContainerMode
    └── enthält eine oder mehrere Edge-Hash-Tabellen
            └── jede verwendet HashGenerator
```

Die drei Container-Modi funktionieren identisch:

```text
DirectHash
    → eine Edge-Tabelle

StaticHash<N>
    → festes Array aus N Edge-Tabellen
    → Auswahl über key[0] % N

DynamicHash
    → dynamisch wachsender Vector
    → Auswahl über key[0] / block_size
```

---

# Kurzes Gesamtbeispiel

```cpp
using Generator = highvoronoi::ExtendedHashGenerator<
    highvoronoi::Murmur128HashGenerator<>,
    highvoronoi::XXHash64<>
>;

using DatabaseConfiguration =
    highvoronoi::DataBaseParams<
        double,
        std::uint32_t,
        Generator,
        highvoronoi::StaticHash<8>,
        highvoronoi::QueueTable
    >;

DatabaseConfiguration database_params{
    highvoronoi::StaticHash<8>{
        1024
    }
};

using EdgeConfiguration =
    highvoronoi::EdgeBufferParams<
        Generator,
        highvoronoi::DynamicHash<>,
        highvoronoi::EdgeTable
    >;

EdgeConfiguration edge_params{
    highvoronoi::DynamicHash<>{
        4,
        10000,
        256
    }
};

highvoronoi::MultiThread threading{8};
```

Das erzeugt effektiv:

```text
Datenbank:

    Scalar = double
    Index  = uint32_t

    festes Array aus 8 Queue-Tabellen
    jede Queue-Tabelle startet mit 1024 Plätzen
    Auswahl der Tabelle durch key[0] % 8
    jede Tabelle verwendet Murmur128 + zusätzlichen XXHash
    jede Tabelle verwendet ReadWriteLock


Edge-Buffer:

    dynamisch wachsender Vector aus Edge-Tabellen
    zunächst 4 Edge-Tabellen
    jede Tabelle startet mit 256 Plätzen

    key[0] =     0 ...  9999  → Tabelle 0
    key[0] = 10000 ... 19999  → Tabelle 1
    usw.

    der Vector wächst bei Bedarf
    jede Edge-Tabelle verwendet denselben HashGenerator
    Vector und Tabellen verwenden ReadWriteLock
```

## Faustregeln

```text
Eine einzelne Tabelle:
    DirectHash

Feste Zahl paralleler Partitionen:
    StaticHash<N>

Unbekannter oder wachsender Bereich von key[0]:
    DynamicHash

Einfache oder alte Referenzimplementierung:
    alte Tabelle

Normale produktive Multi-Thread-Verwendung:
    Tabelle 2

Wenig erwartete Einträge:
    kleine hash_capacity

Viele erwartete Einträge:
    hash_capacity so wählen, dass nicht sofort mehrfach erweitert werden muss
```
