package server

import (
	"archive/zip"
	"bytes"
	"testing"
)

func makeSpreadsheetArchive(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var out bytes.Buffer
	archive := zip.NewWriter(&out)
	for name, content := range files {
		writer, err := archive.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := archive.Close(); err != nil {
		t.Fatal(err)
	}
	return out.Bytes()
}

func TestParseXLSX(t *testing.T) {
	data := makeSpreadsheetArchive(t, map[string]string{
		"xl/workbook.xml":            `<workbook><sheets><sheet name="Tickets" r:id="rId1"/></sheets></workbook>`,
		"xl/_rels/workbook.xml.rels": `<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>`,
		"xl/sharedStrings.xml":       `<sst><si><t>title</t></si><si><t>status</t></si><si><r><t>Broken </t></r><r><t>login</t></r></si><si><t>open</t></si></sst>`,
		"xl/worksheets/sheet1.xml": `<worksheet><sheetData>
<row><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
<row><c r="A2" t="s"><v>2</v></c><c r="B2" t="s"><v>3</v></c></row>
</sheetData></worksheet>`,
	})
	result, err := parseSpreadsheet(data, "tickets.xlsx")
	if err != nil {
		t.Fatal(err)
	}
	if result.Format != "xlsx" || len(result.Items) != 1 || result.Items[0].Fields["title"] != "Broken login" || result.Items[0].Fields["status"] != "open" {
		t.Fatalf("unexpected xlsx result: %#v", result)
	}
}

func TestParseODS(t *testing.T) {
	data := makeSpreadsheetArchive(t, map[string]string{
		"content.xml": `<office:document-content xmlns:office="urn:o" xmlns:table="urn:t" xmlns:text="urn:x"><office:body><office:spreadsheet><table:table table:name="Data"><table:table-row><table:table-cell office:value-type="string"><text:p>name</text:p></table:table-cell></table:table-row><table:table-row><table:table-cell office:value-type="string"><text:p>Ada</text:p></table:table-cell></table:table-row></table:table></office:spreadsheet></office:body></office:document-content>`,
	})
	result, err := parseSpreadsheet(data, "people.ods")
	if err != nil {
		t.Fatal(err)
	}
	if result.Format != "ods" || len(result.Items) != 1 || result.Items[0].Fields["name"] != "Ada" {
		t.Fatalf("unexpected ods result: %#v", result)
	}
}

func TestParseSpreadsheetRejectsLegacyXLS(t *testing.T) {
	if _, err := parseSpreadsheet([]byte("not-a-workbook"), "legacy.xls"); err == nil {
		t.Fatal("legacy xls unexpectedly accepted")
	}
}
