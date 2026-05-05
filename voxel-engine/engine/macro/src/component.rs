use proc_macro::TokenStream;
use quote::quote;
use syn::DeriveInput;

pub fn derive_component_impl(input: TokenStream) -> TokenStream {
    // Convert proc_macro::TokenStream to proc_macro2::TokenStream for internal processing
    let input2: proc_macro2::TokenStream = input.into();

    // Parse using syn::parse2 (works with proc_macro2::TokenStream)
    let DeriveInput { ident, .. } = match syn::parse2::<DeriveInput>(input2) {
        Ok(input) => input,
        Err(e) => return e.to_compile_error().into(),
    };

    let expanded = quote! {
        impl Component for #ident {}
    };

    expanded.into()
}
