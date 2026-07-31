package server

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"path"
	"strconv"
	"strings"
)

const (
	maxSpreadsheetBytes    = 8 << 20
	maxSpreadsheetXMLBytes = 64 << 20
	maxSpreadsheetRows     = 10000
	maxSpreadsheetColumns  = 256
	maxSpreadsheetItems    = 50000
)

type spreadsheetItem struct {
	Fields map[string]string `json:"fields"`
	Text   string            `json:"text"`
	Label  string            `json:"label"`
}

type spreadsheetResponse struct {
	Format  string            `json:"format"`
	Columns []string          `json:"columns"`
	Items   []spreadsheetItem `json:"items"`
}

func parseSpreadsheet(data []byte, filename string) (spreadsheetResponse, error) {
	if len(data) == 0 || len(data) > maxSpreadsheetBytes {
		return spreadsheetResponse{}, fmt.Errorf("spreadsheet must be between 1 byte and %d bytes", maxSpreadsheetBytes)
	}
	ext := strings.ToLower(strings.TrimSpace(filename))
	switch {
	case strings.HasSuffix(ext, ".xlsx"):
		items, columns, err := parseXLSX(data)
		return spreadsheetResponse{Format: "xlsx", Columns: columns, Items: items}, err
	case strings.HasSuffix(ext, ".ods"):
		items, columns, err := parseODS(data)
		return spreadsheetResponse{Format: "ods", Columns: columns, Items: items}, err
	case strings.HasSuffix(ext, ".xls"):
		return spreadsheetResponse{}, errors.New("legacy .xls is not supported; convert it to .xlsx or .ods first")
	default:
		return spreadsheetResponse{}, fmt.Errorf("unsupported spreadsheet extension %q", ext)
	}
}

type xlsxWorkbook struct {
	Sheets []xlsxSheet `xml:"sheets>sheet"`
}

type xlsxSheet struct {
	Name string `xml:"name,attr"`
	RID  string `xml:"id,attr"`
}

type xlsxRelationships struct {
	Relationships []xlsxRelationship `xml:"Relationship"`
}

type xlsxRelationship struct {
	ID     string `xml:"Id,attr"`
	Target string `xml:"Target,attr"`
}

type xlsxSharedStrings struct {
	Items []xlsxSharedString `xml:"si"`
}

type xlsxSharedString struct {
	Text string             `xml:"t"`
	Runs []xlsxSharedString `xml:"r"`
}

type xlsxWorksheet struct {
	Rows []xlsxRow `xml:"sheetData>row"`
}

type xlsxRow struct {
	Cells []xlsxCell `xml:"c"`
}

type xlsxCell struct {
	Ref    string   `xml:"r,attr"`
	Type   string   `xml:"t,attr"`
	Value  string   `xml:"v"`
	Inline []string `xml:"is>t"`
}

func parseXLSX(data []byte) ([]spreadsheetItem, []string, error) {
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, nil, fmt.Errorf("open xlsx archive: %w", err)
	}
	files := make(map[string]*zip.File, len(archive.File))
	for _, file := range archive.File {
		name := path.Clean(strings.TrimPrefix(file.Name, "/"))
		if name == "." || strings.HasPrefix(name, "../") || path.IsAbs(name) {
			continue
		}
		files[name] = file
	}
	workbook, err := decodeZipXML[xlsxWorkbook](files, "xl/workbook.xml")
	if err != nil {
		return nil, nil, err
	}
	rels, err := decodeZipXML[xlsxRelationships](files, "xl/_rels/workbook.xml.rels")
	if err != nil {
		return nil, nil, err
	}
	relTargets := make(map[string]string, len(rels.Relationships))
	for _, rel := range rels.Relationships {
		target := path.Clean(path.Join("xl", rel.Target))
		if strings.HasPrefix(target, "../") || path.IsAbs(target) {
			continue
		}
		relTargets[rel.ID] = target
	}
	shared := []string{}
	if _, ok := files["xl/sharedStrings.xml"]; ok {
		stringsXML, err := decodeZipXML[xlsxSharedStrings](files, "xl/sharedStrings.xml")
		if err != nil {
			return nil, nil, err
		}
		shared = make([]string, len(stringsXML.Items))
		for i, item := range stringsXML.Items {
			shared[i] = item.Text
			for _, run := range item.Runs {
				shared[i] += run.Text
			}
		}
	}
	var items []spreadsheetItem
	var columns []string
	for sheetIndex, sheet := range workbook.Sheets {
		target := relTargets[sheet.RID]
		if target == "" {
			continue
		}
		worksheet, err := decodeZipXML[xlsxWorksheet](files, target)
		if err != nil {
			return nil, nil, fmt.Errorf("read sheet %q: %w", sheet.Name, err)
		}
		rows := make([][]string, 0, len(worksheet.Rows))
		for _, row := range worksheet.Rows {
			cells, err := xlsxCells(row.Cells, shared)
			if err != nil {
				return nil, nil, fmt.Errorf("read sheet %q: %w", sheet.Name, err)
			}
			if len(cells) > 0 {
				rows = append(rows, cells)
			}
			if len(rows) >= maxSpreadsheetRows {
				break
			}
		}
		if len(rows) < 2 {
			continue
		}
		header := uniqueHeaders(rows[0])
		if sheetIndex > 0 || len(workbook.Sheets) > 1 {
			header = append([]string{"sheet"}, header...)
		}
		for _, name := range header {
			if !containsString(columns, name) {
				columns = append(columns, name)
			}
		}
		for rowIndex, row := range rows[1:] {
			fields := make(map[string]string, len(header))
			if len(workbook.Sheets) > 1 {
				fields["sheet"] = sheet.Name
			}
			for i, name := range header {
				if name == "sheet" {
					continue
				}
				if i < len(row) {
					fields[name] = row[i]
				} else {
					fields[name] = ""
				}
			}
			items = append(items, spreadsheetItemFromFields(fields, header, labelForSpreadsheet(fields[header[0]], sheet.Name+" row "+strconv.Itoa(rowIndex+2))))
			if len(items) >= maxSpreadsheetItems {
				return items, columns, nil
			}
		}
	}
	return items, columns, nil
}

func xlsxCells(cells []xlsxCell, shared []string) ([]string, error) {
	values := make([]string, 0, len(cells))
	for index, cell := range cells {
		column := xlsxColumn(cell.Ref)
		if column < 0 {
			column = index
		}
		if column >= maxSpreadsheetColumns {
			continue
		}
		for len(values) <= column {
			values = append(values, "")
		}
		value := cell.Value
		switch cell.Type {
		case "s":
			index, err := strconv.Atoi(strings.TrimSpace(cell.Value))
			if err != nil || index < 0 || index >= len(shared) {
				return nil, fmt.Errorf("invalid shared string index %q", cell.Value)
			}
			value = shared[index]
		case "inlineStr":
			value = strings.Join(cell.Inline, "")
		}
		values[column] = value
	}
	return values, nil
}

func xlsxColumn(ref string) int {
	ref = strings.TrimSpace(ref)
	end := 0
	for end < len(ref) && ((ref[end] >= 'A' && ref[end] <= 'Z') || (ref[end] >= 'a' && ref[end] <= 'z')) {
		end++
	}
	if end == 0 {
		return -1
	}
	column := 0
	for _, char := range strings.ToUpper(ref[:end]) {
		column = column*26 + int(char-'A'+1)
	}
	return column - 1
}

type odsDocument struct {
	Body odsBody `xml:"body"`
}

type odsBody struct {
	Spreadsheet odsSpreadsheet `xml:"spreadsheet"`
}

type odsSpreadsheet struct {
	Tables []odsTable `xml:"table"`
}

type odsTable struct {
	Name string   `xml:"name,attr"`
	Rows []odsRow `xml:"table-row"`
}

type odsRow struct {
	Cells  []odsCell `xml:"table-cell"`
	Repeat int       `xml:"number-rows-repeated,attr"`
}

type odsCell struct {
	Value  string   `xml:"value,attr"`
	String string   `xml:"string-value,attr"`
	Repeat int      `xml:"number-columns-repeated,attr"`
	Text   []string `xml:"p"`
}

func parseODS(data []byte) ([]spreadsheetItem, []string, error) {
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, nil, fmt.Errorf("open ods archive: %w", err)
	}
	files := make(map[string]*zip.File, len(archive.File))
	for _, file := range archive.File {
		files[path.Clean(file.Name)] = file
	}
	document, err := decodeZipXML[odsDocument](files, "content.xml")
	if err != nil {
		return nil, nil, err
	}
	var items []spreadsheetItem
	var columns []string
	for tableIndex, table := range document.Body.Spreadsheet.Tables {
		var rows [][]string
		for _, row := range table.Rows {
			repeat := row.Repeat
			if repeat < 1 {
				repeat = 1
			}
			for repeat > 0 && len(rows) < maxSpreadsheetRows {
				var cells []string
				for _, cell := range row.Cells {
					value := cell.String
					if value == "" {
						value = cell.Value
					}
					if len(cell.Text) > 0 {
						value = strings.Join(cell.Text, " ")
					}
					n := cell.Repeat
					if n < 1 {
						n = 1
					}
					for i := 0; i < n && len(cells) < maxSpreadsheetColumns; i++ {
						cells = append(cells, value)
					}
				}
				if len(cells) > 0 {
					rows = append(rows, cells)
				}
				repeat--
			}
		}
		if len(rows) < 2 {
			continue
		}
		header := uniqueHeaders(rows[0])
		if len(document.Body.Spreadsheet.Tables) > 1 {
			header = append([]string{"sheet"}, header...)
		}
		for _, name := range header {
			if !containsString(columns, name) {
				columns = append(columns, name)
			}
		}
		for rowIndex, row := range rows[1:] {
			fields := make(map[string]string, len(header))
			if len(document.Body.Spreadsheet.Tables) > 1 {
				fields["sheet"] = table.Name
			}
			for i, name := range header {
				if name == "sheet" {
					continue
				}
				if i < len(row) {
					fields[name] = row[i]
				} else {
					fields[name] = ""
				}
			}
			label := table.Name + " row " + strconv.Itoa(rowIndex+2)
			if tableIndex == 0 && len(document.Body.Spreadsheet.Tables) == 1 {
				label = labelForSpreadsheet(fields[header[0]], label)
			}
			items = append(items, spreadsheetItemFromFields(fields, header, label))
			if len(items) >= maxSpreadsheetItems {
				return items, columns, nil
			}
		}
	}
	return items, columns, nil
}

func decodeZipXML[T any](files map[string]*zip.File, name string) (T, error) {
	var result T
	file, ok := files[path.Clean(name)]
	if !ok {
		return result, fmt.Errorf("archive is missing %s", name)
	}
	if file.UncompressedSize64 > maxSpreadsheetXMLBytes {
		return result, fmt.Errorf("archive entry %s is too large", name)
	}
	reader, err := file.Open()
	if err != nil {
		return result, err
	}
	defer reader.Close()
	data, err := io.ReadAll(io.LimitReader(reader, maxSpreadsheetXMLBytes+1))
	if err != nil {
		return result, err
	}
	if len(data) > maxSpreadsheetXMLBytes {
		return result, fmt.Errorf("archive entry %s is too large", name)
	}
	if err := xml.Unmarshal(data, &result); err != nil {
		return result, fmt.Errorf("decode %s: %w", name, err)
	}
	return result, nil
}

func uniqueHeaders(values []string) []string {
	result := make([]string, 0, len(values))
	seen := make(map[string]int, len(values))
	for index, value := range values {
		name := strings.TrimSpace(value)
		if name == "" {
			name = "column" + strconv.Itoa(index+1)
		}
		base := name
		seen[base]++
		if seen[base] > 1 {
			name = base + "_" + strconv.Itoa(seen[base])
		}
		result = append(result, name)
	}
	return result
}

func spreadsheetItemFromFields(fields map[string]string, columns []string, label string) spreadsheetItem {
	parts := make([]string, 0, len(columns))
	for _, column := range columns {
		parts = append(parts, column+": "+fields[column])
	}
	return spreadsheetItem{Fields: fields, Text: strings.Join(parts, "\n"), Label: label}
}

func labelForSpreadsheet(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	if len(value) > 60 {
		return value[:59] + "…"
	}
	return value
}

func containsString(values []string, value string) bool {
	for _, item := range values {
		if item == value {
			return true
		}
	}
	return false
}
